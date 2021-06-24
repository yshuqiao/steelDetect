checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,   #每多少个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
# load_from = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth"
# load_from = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955-e2081918.pth"

# load_from = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927-126fd9bf.pth"

resume_from = None
# resume_from = "/home/yshuqiao/mmdetection/checkpoints/NEU/ga_faster_rcnn_regnetx-3.2GF_1x_gn+ws16/epoch_5.pth"
workflow = [('train', 1)]
