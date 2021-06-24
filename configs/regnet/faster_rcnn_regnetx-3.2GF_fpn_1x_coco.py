_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'   #with groie
]



model = dict(
    pretrained='open-mmlab://regnetx_3.2gf',
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_3.2gf',
        out_indices=(0, 1, 2, 3),

        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
# norm_cfg = dict(type='GN', num_groups=16, requires_grad=True),
# conv_cfg = dict(type='ConvWS'),
        norm_eval=True,
        style='pytorch',

        # se=True,  # I add this
        # cbam=True,

        # add empirical attention
#         plugins=[
#             dict(
#                 cfg=dict(
#                     type='GeneralizedAttention',
#                     spatial_range=-1,
#                     num_heads=8,
#                     attention_type='1111',
#                     kv_stride=2),
#                 stages=(False, False, True, True),
# # stages=(False, True, True, True),
#                 position='after_conv2')
#         ],

# plugins=[
#         dict(
#             cfg=dict(
#                 type='GeneralizedAttention',
#                 spatial_range=-1,
#                 num_heads=8,
#                 attention_type='0010',
#                 kv_stride=2),
#             stages=(False, False,  True, True),
#             position='after_conv2')
#     ],

# add dcn
#         dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
#         stage_with_dcn=(False, True, True, True)


    ),
    neck=dict(
# _delete_=True,
        type='FPN',
        # type='AUGFPN',
# type='HRFPN',
        in_channels=[96, 192, 432, 1008],
        out_channels=256,

# target_size_list=[8, 16, 32, 64, 128],  # add this
        num_outs=5,
# num_outs=4,
# norm_cfg = dict(type='GN', num_groups=16, requires_grad=True),
# conv_cfg = dict(type='ConvWS'),
    ),

    # neck=dict(
    #     type='FPN_CARAFE',
    #     # in_channels=[256, 512, 1024, 2048],
    #     in_channels=[96, 192, 432, 1008],
    #     out_channels=256,
    #     num_outs=5,
    #     start_level=0,
    #     end_level=-1,
    #     norm_cfg=None,
    #     act_cfg=None,
    #     order=('conv', 'norm', 'act'),
    #     upsample_cfg=dict(
    #         type='carafe',
    #         up_kernel=5,
    #         up_group=1,
    #         encoder_kernel=3,
    #         encoder_dilation=1,
    #         compressed_channels=48))

)


img_norm_cfg = dict(
    # The mean and std are used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(200, 200), keep_ratio=True),
# dict(type='Resize', img_scale=(600, 600), keep_ratio=True),
# dict(
#         type='Resize',
#         img_scale=[(200, 160), (200, 168), (200, 176), (200, 184),
#                    (200, 192), (200, 200)],
#         multiscale_mode='value',
#         keep_ratio=True),
# dict(
#         type='Resize',
#         img_scale=[(600, 480), (600, 504), (600, 528), (600, 552),
#                    (600, 576), (600, 600)],
#         multiscale_mode='value',
#         keep_ratio=True),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(200, 200),
# img_scale=(600, 600),  # notice here
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00005)
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.00005)
