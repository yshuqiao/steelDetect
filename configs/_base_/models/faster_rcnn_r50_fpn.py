model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        # type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',

        # # add empirical attention
        # plugins=[
        #     dict(
        #         cfg=dict(
        #             type='GeneralizedAttention',
        #             spatial_range=-1,
        #             num_heads=8,
        #             attention_type='1111',
        #             kv_stride=2),
        #         stages=(False, False, True, True),
        #         position='after_conv2')
        # ],

        # add dcn
        # dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
# type='BIFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
# target_size_list=[8, 16, 32, 64, 128],  # add this
        num_outs=5),
# num_outs=4),

    # neck=dict(
    #     type='FPN_CARAFE',
    #     in_channels=[256, 512, 1024, 2048],
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
    #         compressed_channels=64)),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],   # 与strides相乘得到anchor面积，再由ratios决定长宽
            # scales=[4],
            # scales=[2],
            ratios=[0.5, 1.0, 2.0],
            # ratios=[0.2, 0.5, 1.0, 2.0],
            # ratios=[0.1, 0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),  # 与scales乘得到anchor平均边长，即8*4,8*8,...,8*64
# strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(

        type='StandardRoIHead',

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # type='SoftRoIExtractor',  # notice
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),


        bbox_head=dict(  # notice here
            type='Shared2FCBBoxHead',
# type='Shared4Conv1FCBBoxHead',
# conv_out_channels=256,
# norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
# conv_cfg=dict(type='ConvWS'),
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            # num_classes=80,
            num_classes=6,
# num_classes=5,
            # num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),

            reg_class_agnostic=False,

            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  # could be revise to FocalLoss
            # loss_cls=dict(
            # type='FocalLoss',
            # use_sigmoid=True,
            # gamma=2.0,
            # alpha=0.25,
            # loss_weight=1.0),

            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
# loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
# reg_decoded_bbox=True,
# loss_bbox=dict(type='BoundedIoULoss', loss_weight=10.0)
# loss_bbox=dict(type='CIoULoss', loss_weight=10.0)
        )
    )


)

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',  # here could be revise to OHEMSampler
            # type='OHEMSampler',
# type='InstanceBalancedPosSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),

        pos_weight=-1,

        # use_consistent_supervision=True,  # I add this

        debug=False,

    ))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),  # here could be revise to soft_nms
        # nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),  # attention1111fftt
        max_per_img=100,

        # use_consistent_supervision=True,  # I add this

    )
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)


