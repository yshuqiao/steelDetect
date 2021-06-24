import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmcv.cnn import constant_init,kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
import functools

from mmcv.runner import auto_fp16

from mmcv.runner import force_fp32
from mmcv.cnn import build_norm_layer
import torch

from ..builder import NECKS

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)

class SeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=False):
        super(SeparableConv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = relu

        self.sep = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=bias
        )
        if relu:
            self.relu_fn = Swish()

    def forward(self, x):
        x = self.pw(self.sep(x))
        if self.relu:
            x = self.relu_fn(x)
        return x

class WeightedInputConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ins,
                 conv_cfg=None,
                 norm_cfg=None,
                 separable_conv=True,
                 act_cfg=None,
                 eps=0.0001):
        super(WeightedInputConv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_ins = num_ins
        self.eps = eps
        self.separable_conv = separable_conv

        self.sep_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False
        )
        self.pw_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            inplace=False
        )

        # td_conv = torch.nn.Sequential(td_sep_conv,td_pw_conv)
        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(1.0))
        self.relu = nn.ReLU(inplace=False)
        # self.relu = F.relu

    def forward(self,inputs):
        assert isinstance(inputs,list)
        assert len(inputs) == self.num_ins
        w = self.relu(self.weight)
        w /= (w.sum()+self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i]*inputs[i]
        output = self.pw_conv(self.sep_conv(F.relu(x)))
        return output

class WeightedInputConv_V2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ins,
                 conv_cfg=None,
                 norm_cfg=None,
                 separable_conv=True,
                 act_cfg=None,
                 eps=0.0001):
        super(WeightedInputConv_V2,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_ins = num_ins
        self.eps = eps

        if separable_conv:
            _,bn_layer = build_norm_layer(norm_cfg,out_channels)
            self.conv_op = nn.Sequential(
                SeparableConv(
                    in_channels,
                    out_channels,
                    bias=True,
                    relu=False
                ),
                bn_layer
            )
        else:
            self.conv_op = ConvModule(
                in_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False
            )

        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(1.0))
        self._swish = Swish()

    def forward(self,inputs):
        assert isinstance(inputs,list)
        assert len(inputs) == self.num_ins
        w = F.relu(self.weight)
        w /= (w.sum()+self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i]*inputs[i]
        # import pdb;pdb.set_trace()
        output = self.conv_op(self._swish(x))
        return output

class ResampingConv(nn.Module):
    """
        in_channels,
        in_width,
        target_width,
        target_num_channels,
        conv_cfg=None,
        norm_cfg=None,
        separable_conv=False,
        act_cfg=None
        """
    def __init__(self,
                 in_channels,
                 in_stride,
                 out_stride,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 separable_conv=False,
                 act_cfg=None
                 ):
        super(ResampingConv,self).__init__()
        # assert out_stride % in_stride == 0 or out_stride % in_stride == 0
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.out_stride = out_stride
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if self.in_stride < self.out_stride:
            scale = int(self.out_stride//self.in_stride)
            assert scale == 2
            self.rescale_op = nn.MaxPool2d(
                scale+1,
                stride=scale,
                padding=1
            )
            # self.rescale_op = nn.MaxPool2d(2, stride=scale)
        else:
            if self.in_stride > self.out_stride:
                scale = self.in_stride // self.out_stride
                self.rescale_op = functools.partial(
                    F.interpolate,scale_factor=scale,mode='nearest'
                )
            else:
                self.rescale_op = None

        if self.in_channels != self.out_channels:
            if separable_conv:
                raise NotImplementedError
            else:
                self.conv_op = ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                    inplace=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        if self.in_channels != self.out_channels:
            x = self.conv_op(x)
        x = self.rescale_op(x) if self.rescale_op else x
        return x

class bifpn(nn.Module):
    nodes_settings = [
        {'width_ratio': 64, 'inputs_offsets': [3, 4]},
        {'width_ratio': 32, 'inputs_offsets': [2, 5]},
        {'width_ratio': 16, 'inputs_offsets': [1, 6]},
        {'width_ratio': 8, 'inputs_offsets': [0, 7]},
        {'width_ratio': 16, 'inputs_offsets': [1, 7, 8]},
        {'width_ratio': 32, 'inputs_offsets': [2, 6, 9]},
        {'width_ratio': 64, 'inputs_offsets': [3, 5, 10]},
        {'width_ratio': 128, 'inputs_offsets': [4, 11]},
    ]

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=[8, 16, 32, 64, 128],
                 num_outs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_batch_norm=False,
                 act_cfg=None
                 ):
        super(bifpn, self).__init__()
        assert num_outs>=2
        assert len(strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_outs = num_outs

        self.channels_nodes = [i for i in in_channels]
        self.stride_nodes = [i for i in strides]
        self.resample_op_nodes = nn.ModuleList()
        self.new_op_nodes = nn.ModuleList()

        for _,fnode in enumerate(self.nodes_settings):
            new_node_stride = fnode['width_ratio']
            op_node = nn.ModuleList()
            for _,input_offset in enumerate(fnode['inputs_offsets']):
                input_node = ResampingConv(
                    self.channels_nodes[input_offset],
                    self.stride_nodes[input_offset],
                    new_node_stride,
                    out_channels,
                    norm_cfg=norm_cfg
                )
                op_node.append(input_node)
            new_op_node = WeightedInputConv_V2(
                out_channels,
                out_channels,
                len(fnode['inputs_offsets']),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.new_op_nodes.append(new_op_node)
            self.resample_op_nodes.append(op_node)
            self.channels_nodes.append(out_channels)
            self.stride_nodes.append(new_node_stride)

    def forward(self,inputs):
        assert len(inputs) == self.num_outs
        feats = [i for i in inputs]
        for fnode,op_node,new_op_node in zip(self.nodes_settings,
                                               self.resample_op_nodes, self.new_op_nodes):
            input_node = []
            for input_offset,resample_op in zip(fnode['inputs_offsets'], op_node):
                input_node.append(resample_op(feats[input_offset]))
            feats.append(new_op_node(input_node))
            # add hist

        outputs = feats[-self.num_outs:]
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
                 num_outs,
                 strides = [8,16,32,64,128],  # add this
                 # strides=[8, 16, 32, 64],
                 # strides=[7, 14, 28, 56],
                 start_level=0,
                 end_level=-1,
                 # add_extra_convs=False,
                 # extra_convs_on_inputs=True,
                 # relu_before_extra_convs=False,
                 # no_norm_on_lateral=False,
                 stack=3,  # add this
                 conv_cfg=None,
                 # norm_cfg=None,
                 # norm_cfg=dict(type='SyncBN', momentum=0.01,
                 #            eps=1e-3, requires_grad=True),
                 norm_cfg=dict(type='BN', momentum=0.01,
                               eps=1e-3, requires_grad=True),
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        # super(FPN, self).__init__()
        super(BIFPN, self).__init__()
        # assert isinstance(in_channels, list)
        assert len(in_channels) >= 3  # add this
        assert len(strides) == len(in_channels)  # add this
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.strides = strides

        self.num_ins = len(in_channels)

        self.act_cfg = act_cfg  # add this
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

        bifpn_in_channels = in_channels[self.start_level:self.backbone_end_level]
        bifpn_strides = strides[self.start_level:self.backbone_end_level]
        bifpn_num_outs = self.num_outs

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
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
        self.extra_convs = None
        if extra_levels>=1:
            self.extra_convs = nn.ModuleList()
            for _ in range(extra_levels):
                self.extra_convs.append(
                    ResampingConv(
                        bifpn_in_channels[-1],
                        bifpn_strides[-1],
                        bifpn_strides[-1]*2,
                        out_channels,
                        norm_cfg=norm_cfg
                    )
                )
                bifpn_in_channels.append(out_channels)
                bifpn_strides.append(bifpn_strides[-1]*2)

        self.stack_bifpns = nn.ModuleList()
        for _ in range(stack):
            self.stack_bifpns.append(
                bifpn(
                    bifpn_in_channels,
                    out_channels,
                    strides=bifpn_strides,
                    num_outs=bifpn_num_outs,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    act_cfg=None
                )
            )
            # import pdb; pdb.set_trace()
            bifpn_in_channels = [out_channels for _ in range(bifpn_num_outs)]




    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

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

        feats = list(inputs[self.start_level:self.backbone_end_level])
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                feats.append(self.extra_convs[i](feats[-1]))
        for idx,stack_bifpn in enumerate(self.stack_bifpns):
            feats = stack_bifpn(feats)
        # return tuple(x)
        return tuple(feats[:self.num_outs])

        # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, 1)
