### mmdetection原本工具使用
- 此项目建立在<https://github.com/open-mmlab/mmdetection>的基础上，其中<i>标题Getting Started</i>下有些链接比较概括
- 首先按这个网址的方法进行安装<https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md>
- 然后可以看一下官方的这个使用教程，里面也有些链接<https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md>
- 主要是用`python tools/train.py config_path --work_dir checkpoint_save_path`来进行训练
- 可以用test.py来获取<b>mAP指标</b>，即`python tools/test.py config_path checkpoint_path --eval bbox`
- 画loss曲线图可以用`python tools/analyze_logs.py plot_curve xxx.log.json --keys loss --legend loss`
- 获取网络的浮点运算量可以用`python tools/get_flops.py config_path --shape 200`
- `python tools/coco_error_analysis.py result.json output_dir --ann GT.json`可以用来分析误差，
<br>test.py如果设置--out参数则一般是通过single_gpu_test生成后缀为.pkl的pickle文件，但根据里面
    ```
    if args.format_only:
           dataset.format_results(outputs,**kwargs)
    ```
    这段代码，可以看到mmdetection/mmdet/datasets/coco.py文件的evaluate函数可以通过`result_files, tmp_dir = self.format_results(results, jsonfile_prefix) `来生成.json后缀的结果，
    <br>那么运行`python tools/test.py config_path checkpoint_path --format-only --eval-options jsonfile_prefix = "/home/yshuqiao/mmdetection/tools/results/regnet"`则可生成后缀为.json的结果文件regnet.json从而可以用coco_error_analysis.py来分析误差。

### tools文件夹下另增的脚本
- coco2GTjson.py可以用来把<font face="黑体" color=#008000 size=5>coco格式标注文件</font>转化为GTjson文件，后续可以用*visualizeGTandPredict.py*来可视化图片上的真实标注框。
- defect_nums.py可以统计<font face="黑体" color=green size=5>coco格式标注文件</font>中各类缺陷的个数。
- inference-me.py可以用来生成**结果的json文件**，后续可用*visualizeGTandPredict.py*来可视化图片上的预测框。

### 在mmdetection基础上修改的地方
|resnet.py|regnet.py|schedule_1x.py|
|:------|------:|:-------:|
|se|cbam|cosineAnnealing|

1.resnet.py中增加SE通道注意力
<br>主要是在resnet.py里面添加SELayer类：
```python
class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)
```
并在Bottleneck模块类中添加self.se参数，在其中的forward函数里添加：
```python
if self.se:
    se_block = SELayer(self.planes)
    out = se_block(out)
```
Bottleneck类中也能看到，如果加入generalized_attention模块（Transformer）这样的插件的话，会插入到Bottleneck中第一个卷积层conv1（卷积核大小为3x3，紧接着BN层和relu）和第二个卷积层conv2之间。

2.regnet.py中增加CBAM混合注意力
<br>在类中添加self.se参数（其Bottleneck类会调用resnext.py中的Bottleneck，而resnext.py中的Bottleneck继承了resnet.py的Bottleneck）和self.cbam参数，
<br>如果self.cbam为True，则初始化ChannelAttention类和SpatialAttention类，相应地，在前传函数forward中的最后一个阶段后面添加cbam。
<br>而如果self.se为True，则对第三阶段和第四阶段的残差层传入self.se=True的参数值，使残差层中每一个Bottleneck的残差支路末尾，即在与恒等映射支路相加之前添加SE模块。
3.把schedule_1x.py中的step学习率衰减策略改为cosineAnnealing学习率衰减策略。
<br>根据<https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_runtime.md>，即把
```python
lr_config = dict(
    policy='step',
    warmup='linear',  # 可改为余弦退火算法
    warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.001,  # 起始的学习率
    step=[8, 11]  # 在第8和11个epoch时降低学习率，可调成step=[16, 19]
)
```
改为
```python
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0/10,
    min_lr_ratio=1e-5,
)
```

### 其他
- 模型可视化工具[Netron](https://github.com/lutzroeder/Netron)
- 对mmdetection的解读：[ GiantPandaCV公众号中【MMDetection 超全专栏】][1]、[OpenMMLab | 轻松掌握 MMDetection 整体构建流程(一)][2]、[MMDETECTION最小复刻版(五)：YOLOV5转化内幕][3]
- 根据需要调整config文件（以faster_rcnn_r50_fpn_1x_coco.py为例）：
   - coco格式数据的制作与配置修改：对后缀为xml的voc格式的数据标签，可以通过脚本（可查找CSDN博客相关参考）转换成后缀为json的coco格式的；若自己制作数据集，可用labelimg或labelme，也是可以通过脚本转化成coco格式；
   <br>可以通过对数据标签的统计分析得到数据集目标的数量、长宽比、面积分布等信息；通过脚本划分数据集为训练集和测试集，对训练集的数据增强可以参考<https://github.com/maozezhong/CV_ToolBox>；
   <br>配置文件中，需要调整coco_detection.py中的data_root,img_scale,samples_per_gpu(batchsize)；mmdet/datasets/coco.py里面的类别名称CLASSES，另外可设iou_thrs=[5]。
   - 学习率设置与否使用预训练模型：schedule_1x.py中如果优化器用了SGD，那么一般设置学习率lr=gpu_nums·0.00125·samples_per_gpu；default_runtime.py中可以设置load_from=预训练模型，其中预训练模型可以由项目首页的**Benchmark and model zoo**标题中点击Faster R-CNN，跳转到<https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn>选择下载。
   - 目标类别数、显示各类mAP、tensorboard的使用：faster_rcnn_r50_fpn.py中要把num_classes设置成目标类别数；如果要验证模型效果的时候显示各类目标的mAP，那么要在mmdet/datasets/coco.py的evaluate函数中设置classwise=True；
   <br>若要使用tensorboard查看模型训练过程中loss变化，应在default_runtime.py中去掉`dict(type='TensorboardLoggerHook')`的注释，并安装tensorboard包，然后在命令终端中输入`tensorboard --logdir=work_dir/`，就可以在浏览器打开http://localhost:6006/查看。
   - 模型其他配置参数的调整或替换：anchor_generator的scales,ratios和strides；img_norm_cfg中的mean和std；骨干网络或特征金字塔的替换；train_cfg里的rcnn里的sampler的type可用改为OHEMSampler；
   <br>roi_head里面loss_bbox可以改为CIoULoss并设置`reg_decoded_bbox=True`，loss_cls可以改为FocalLoss，其参数设置为：
       ```python
       loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
       ```
       配置文件中单引号的名称可以copy来在pycharm此项目中搜索，例如'CrossEntropyLoss'和'FocalLoss'，能找到相应的定义且可能找到可替换的部件。
   
   - **mmdetection**重点关注的文件夹是configs（含各种配置文件）、mmdet（含各种模型组件）和tools（含训练测试等主要工具），
   <br>大致可以认为mmdetection中有一个*大管家*，tools中的脚本是主函数，通过把config配置文件中的参数读到给*大管家*，把数据和模型（mmdet提供模型结构等）放到*大管家*；*大管家*用统一训练测试管道输出评价结果；
   <br>可以重点关注这些py文件：mmdet/models/detectors/two_stage.py、mmdet/models/roi_heads/standard_roi_heads.py、mmdet/models/dense_heads/rpn_head.py以及
   <br>训练信息相关:从tools/train.py文件ctrl点击train_detector可进入mmdet/apis/train.py，看到tran_detector函数底下有`runner.run(data_loaders, cfg.workflow, cfg.total_epochs)`,
   <br>再Ctrl点击run可进入所安装的包的内部文件/home/yshuqiao/anaconda/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py，
   <br>其中可以看到run函数下面有`epoch_runner(data_loaders[i],**kargs)`实则调用了train函数中的`self.run_iter(data_batch,train_mode=True)`，另外还用到了`logger.info`，但是找不到打印训练的loss和acc等信息的输出处，
   <br>倒是只在mmdet/datasets/coco.py中看到`print_log(log_msg, logger=logger)`，或许是epoch_based_runner.py中的val函数通过run_iter中的val_step间接地调用了tests/test_data/test_dataset.py中的test_dataset_evaluation函数或是tools/test.py或tools/eval_metric.py,
   <br>很大可能是与tests/test_data/test_dataset.py中_build_demo_runner的val_step以及test_evaluation_hook函数有关，而test_evaluation_hook函数中`evalhook.evaluate = MagicMock()`Ctrl点击evaluate可以进入mmdet/core/evaluation/eval_hooks.py中的evaluate函数，
   <br>其中`eval_res = self.dataloader.dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)`中evaluate对应coco的则为coco_dataset.evaluate，而对应自己定义的数据集则调用相应的evaluate函数；另外在mmdet/apis/train.py中还有这一句`eval_hook = DistEvalHook if distributed else EvalHook`，Ctrl点击EvalHook也能跳转到mmdet/core/evaluation/eval_hooks.py。
   <br>至于不同版本的mmdetection(tags)，可以在github项目左上角下拉选择；ctrl+F可以在指定github项目中搜索关键字，方便查找一些模型结构相关的py文件。
   - 关于深度学习说两句：机器学习中主要了解矩阵、概率等相关概念；卷积神经网络中可以把卷积核与滤波器联系起来（[吴恩达的深度学习教程][4]b站上有）；其他要学习了解Linux、Cuda、Anaconda、pytorch的操作使用。
   - 除了[faster RCNN](https://zhuanlan.zhihu.com/p/137454940?utm_source=wechat_session&utm_medium=social&utm_oi=57622111715328)，yolov5、efficientdet、fcos、centernet可以去了解一下，其中fcos使用了atss自适应采样，centernet是无锚框的检测算法；
   <br>跑通一个网络主要关注：*数据的预处理和结果评价*、*模型结构*、*优化器配置*等，看代码过程中除了debug，可用print来看各个环节输出的张量形状。

[1]:https://mp.weixin.qq.com/s/LslHGseIj8fO001Vf0C9pA
[2]:https://mp.weixin.qq.com/s/vcFH5gChIYGxFeFxuwEzAw
[3]:https://www.freesion.com/article/87631393267/
[4]:http://www.bilibili.com/video/BV18J411R7dT?p=1&share_medium=android&share_source=qq&bbid=XY64998611229D67449BCE373319CE1C58780&ts=1624438367556