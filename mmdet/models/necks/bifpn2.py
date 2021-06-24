import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

import torch

from ..builder import NECKS

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        result = i*torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx,grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output*(sigmoid_i * (1+i*(1-sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self,x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=dict(type='BN',momentum=0.003,eps=1e-4,requires_grad=True),
                 activation=None,
                 bias=False
                 ):
        super(SeparableConv2d,self).__init__()
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,
                                   stride,padding,dilation,groups=in_channels,bias=False)
        self.pointwise = ConvModule(in_channels,out_channels,1,norm_cfg=norm_cfg,act_cfg=None,bias=bias,inplace=False)
        if activation=="ReLU":
            self.act = nn.ReLU()
        elif activation=="Swish":
            self.act = MemoryEfficientSwish()
            # self.act = Swish()
        else:
            self.act = None

    def init_weights(self):
        xavier_init(self.depthwise,distribution='uniform')
        xavier_init(self.pointwise.conv, distribution='uniform')

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.act:
            x = self.act(x)
        return x

class WeightedMerge(nn.Module):
    def __init__(self,in_channels,out_channels,target_size,norm_cfg,apply_bn=False,eps=0.0001):
        super(WeightedMerge,self).__init__()
        self.conv = SeparableConv2d(out_channels,out_channels,3,padding=1,norm_cfg=norm_cfg,bias=True)
        self.eps = eps
        self.num_ins = len(in_channels)
        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(1))
        self.relu = nn.ReLU(inplace=False)
        self.swish = MemoryEfficientSwish()
        # self.swish = Swish()
        self.resample_ops = nn.ModuleList()
        for in_c in in_channels:
            self.resample_ops.append(Resample(in_c,out_channels,target_size,norm_cfg,apply_bn))

    def forward(self,inputs):
        assert isinstance(inputs,list)
        assert len(inputs) == self.num_ins
        w = self.relu(self.weight)
        w /= (w.sum()+self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i]*self.resample_ops[i](inputs[i])
        output = self.conv(self.swish(x))
        return output

class Resample(nn.Module):
    def __init__(self,in_channels,out_channels,target_size,norm_cfg,apply_bn=False):
        super(Resample,self).__init__()
        self.target_size = torch.Size([target_size,target_size])
        self.is_conv = in_channels!=out_channels
        if self.is_conv:
            self.conv = ConvModule(in_channels,
                                   out_channels,
                                   1,
                                   norm_cfg=norm_cfg if apply_bn else None,
                                   bias=True,
                                   act_cfg=None,
                                   inplace=False)
    def _resize(self,x,size):
        if x.shape[-2:]==size:
            return x
        elif x.shape[-2:]<size:
            return F.interpolate(x,size=size,mode='nearest')
        else:
            assert x.shape[-2]%size[-2]==0 and x.shape[-1]%size[-1]==0
            kernel_size = x.shape[-1]//size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size+1, stride=kernel_size, padding=1)
            return x
    def forward(self,inputs):
        if self.is_conv:
            inputs = self.conv(inputs)
        return self._resize(inputs,self.target_size)

class bifpn_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 target_size_list,
                 num_outs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(bifpn_layer, self).__init__()
        assert num_outs >= 2
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.num_outs = num_outs

        self.top_down_merge = nn.ModuleList()
        for i in range(self.num_outs-1,0,-1):
            in_channels_list = [out_channels,in_channels[i-1]] if i<self.num_outs-1 else [in_channels[i],in_channels[i-1]]
            merge_op = WeightedMerge(in_channels_list,out_channels,target_size_list[i-1],norm_cfg,apply_bn=True)
            self.top_down_merge.append(merge_op)

        self.bottom_up_merge = nn.ModuleList()
        for i in range(0,self.num_outs-1):
            in_channels_list = [out_channels,in_channels[i+1],out_channels] if i<self.num_outs-2 else [in_channels[-1],out_channels]
            merge_op = WeightedMerge(in_channels_list, out_channels, target_size_list[i + 1], norm_cfg, apply_bn=True)
            self.bottom_up_merge.append(merge_op)

    def forward(self,inputs):
        assert len(inputs) == self.num_outs

        # top down merge
        md_x = []
        for i in range(self.num_outs-1,0,-1):
            inputs_list = [md_x[-1],inputs[i-1]] if i<self.num_outs-1 else [inputs[i],inputs[i-1]]
            x = self.top_down_merge[self.num_outs-i-1](inputs_list)
            md_x.append(x)

        # bottom up merge
        outputs = md_x[::-1]
        for i in range(1,self.num_outs-1):
            outputs[i] = self.bottom_up_merge[i-1]([outputs[i],inputs[i],outputs[i-1]])
        outputs.append(self.bottom_up_merge[-1]([inputs[-1],outputs[-1]]))
        return outputs




@NECKS.register_module()
# class FPN(nn.Module):
class BIFPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,

                 target_size_list,

                 num_outs,
                 start_level=0,
                 end_level=-1,
                 # add_extra_convs=False,
                 # extra_convs_on_inputs=True,
                 # relu_before_extra_convs=False,
                 # no_norm_on_lateral=False,

                 stack=1,

                 conv_cfg=None,
                 # norm_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.003, eps=1e-4, requires_grad=True),

                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        # super(FPN, self).__init__()
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)

        assert len(in_channels) >= 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.stack = stack

        self.num_outs = num_outs
        # self.relu_before_extra_convs = relu_before_extra_convs
        # self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        # self.add_extra_convs = add_extra_convs
        # assert isinstance(add_extra_convs, (str, bool))
        # if isinstance(add_extra_convs, str):
        #     # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
        #     assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        # elif add_extra_convs:  # True
        #     if extra_convs_on_inputs:
        #         # For compatibility with previous release
        #         # TODO: deprecate `extra_convs_on_inputs`
        #         self.add_extra_convs = 'on_input'
        #     else:
        #         self.add_extra_convs = 'on_output'

        # self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()
        #
        # for i in range(self.start_level, self.backbone_end_level):
        #     l_conv = ConvModule(
        #         in_channels[i],
        #         out_channels,
        #         1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
        #         act_cfg=act_cfg,
        #         inplace=False)
        #     fpn_conv = ConvModule(
        #         out_channels,
        #         out_channels,
        #         3,
        #         padding=1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #         inplace=False)
        #
        #     self.lateral_convs.append(l_conv)
        #     self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        self.extra_ops = nn.ModuleList()
        for i in range(self.backbone_end_level,self.num_outs):
            in_c = in_channels[-1]
            self.extra_ops.append(
                Resample(in_c,out_channels,target_size_list[i],norm_cfg,apply_bn=True)
            )
            in_channels.append(out_channels)

        self.stack_bifpns = nn.ModuleList()
        for _ in range(stack):
            self.stack_bifpns.append(
                bifpn_layer(in_channels,
                            out_channels,
                            target_size_list,
                            num_outs=self.num_outs,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg))
            in_channels = [out_channels] * self.num_outs

        # extra_levels = num_outs - self.backbone_end_level + self.start_level
        # if self.add_extra_convs and extra_levels >= 1:
        #     for i in range(extra_levels):
        #         if i == 0 and self.add_extra_convs == 'on_input':
        #             in_channels = self.in_channels[self.backbone_end_level - 1]
        #         else:
        #             in_channels = out_channels
        #         extra_fpn_conv = ConvModule(
        #             in_channels,
        #             out_channels,
        #             3,
        #             stride=2,
        #             padding=1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             inplace=False)
        #         self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     xavier_init(m, distribution='uniform')
            if isinstance(m, SeparableConv2d):
                m.init_weights()


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # assert len(inputs) == len(self.in_channels)
        #
        # # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        #
        # # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #     #  it cannot co-exist with `size` in `F.interpolate`.
        #     if 'scale_factor' in self.upsample_cfg:
        #         laterals[i - 1] += F.interpolate(laterals[i],
        #                                          **self.upsample_cfg)
        #     else:
        #         prev_shape = laterals[i - 1].shape[2:]
        #         laterals[i - 1] += F.interpolate(
        #             laterals[i], size=prev_shape, **self.upsample_cfg)
        #
        # # build outputs
        # # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        # # part 2: add extra levels
        # if self.num_outs > len(outs):
        #     # use max pool to get more levels on top of outputs
        #     # (e.g., Faster R-CNN, Mask R-CNN)
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs - used_backbone_levels):
        #             outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        #     # add conv layers on top of original feature maps (RetinaNet)
        #     else:
        #         if self.add_extra_convs == 'on_input':
        #             extra_source = inputs[self.backbone_end_level - 1]
        #         elif self.add_extra_convs == 'on_lateral':
        #             extra_source = laterals[-1]
        #         elif self.add_extra_convs == 'on_output':
        #             extra_source = outs[-1]
        #         else:
        #             raise NotImplementedError
        #         outs.append(self.fpn_convs[used_backbone_levels](extra_source))
        #         for i in range(used_backbone_levels + 1, self.num_outs):
        #             if self.relu_before_extra_convs:
        #                 outs.append(self.fpn_convs[i](F.relu(outs[-1])))
        #             else:
        #                 outs.append(self.fpn_convs[i](outs[-1]))
        # return tuple(outs)

        outs = list(inputs)
        for _,extra_op in enumerate(self.extra_ops):
            outs.append(extra_op(outs[-1]))

        for _,stack_bifpn in enumerate(self.stack_bifpns):
            outs = stack_bifpn(outs)

        return tuple(outs[:self.num_outs])
