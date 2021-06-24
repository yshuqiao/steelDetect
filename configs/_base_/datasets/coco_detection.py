dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
# data_root = "/home/yshuqiao/datas/coco/"
# data_root = "/home/yshuqiao/datas/NEU-DET4-fix-random/"
# data_root = "/home/yshuqiao/datas/NEU-DET5-fix/"
data_root = "/home/yshuqiao/datas/NEU-DET/"
# data_root = "/home/yshuqiao/datas/NEU-DET-newCrazing2/"
# data_root = "/home/yshuqiao/datas/NEU-DET-NoCrazing/"
# data_root = "/home/yshuqiao/datas/NEU-DET-augs/"
# data_root = "/home/yshuqiao/datas/NEU-DET-augs0.8/"
# data_root = "/home/yshuqiao/datas/NEU-DET-augs0.8seed100/"
# data_root = "/home/yshuqiao/datas/severstal-steel-detect/"

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 貌似人为设定
img_norm_cfg = dict(
    mean=[128.285, 128.285, 128.285], std=[44.525, 44.525, 44.525], to_rgb=True)

# img_norm_cfg = dict(
#     # The mean and std are used in PyCls when training RegNets
#     mean=[103.53, 116.28, 123.675],
#     std=[57.375, 57.12, 58.395],
#     to_rgb=False)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(200, 200), keep_ratio=True),  # 这里可以多尺度，如img_scale=[(200,160), (200,200)],multiscale_mode='range',
    # dict(type='Resize', img_scale=(1600, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(200, 200),
        # img_scale=(1600, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,   # notice here
# samples_per_gpu=8,
    workers_per_gpu=1,   # notice here
# workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',

        # ann_file=data_root + 'annotations/train_iron.json',  # original
# ann_file=data_root + 'trainNew_iron.json',
#         img_prefix=data_root + 'images_divide/train/',   # original
# img_prefix=data_root + 'trainBright/',
# img_prefix=data_root + 'trainClahe/',

# ann_file=[data_root + 'annotations/train_iron.json', data_root + 'trainMosaic.json'],
# img_prefix=[data_root + 'images_divide/train/', data_root + 'mosaic/'],

# ann_file=[data_root + 'annotations/train_iron.json', data_root + 'trainRotate_iron.json', data_root + 'crazingHSV_iron.json', data_root + 'rollHSV_iron.json'],
#         img_prefix=[data_root + 'images_divide/train/', data_root + 'trainImg_rotate/', data_root + 'crazingHSVImg/', data_root + 'rollHSVImg/'],
# ann_file=[data_root + 'annotations/train_iron.json', data_root + 'crazing3_iron.json'],
#         img_prefix=[data_root + 'images_divide/train/', data_root + 'crazing_v3_img/'],
        # ann_file=[data_root + 'train_iron.json', data_root + 'trainCrop_iron.json', data_root + 'trainRotate_iron.json'],
        # img_prefix=[data_root + 'trainImg/', data_root + 'trainImg_crop/', data_root + 'trainImg_rotate/'],
# ann_file=data_root + 'train_iron.json',
#         img_prefix=data_root + 'trainImg/',

# this crop augs
ann_file=[data_root + 'annotations/train_iron.json', data_root + 'trainCrop_iron.json'],  # augs
img_prefix=[
            '/home/yshuqiao/datas/NEU-DET/images_divide/train/',
            '/home/yshuqiao/datas/NEU-DET/trainImg_crop/'
        ],

# ann_file=[data_root + 'annotations/train_iron.json', data_root + 'trainCrop_iron.json', data_root + 'rotate45_iron.json'],
#         img_prefix=[data_root + 'images_divide/train/', data_root + 'trainImg_crop/', data_root + 'trainImg_rotate45/'],

        # ann_file=data_root + 'severstal_annotations/train_steel.json',
        # img_prefix=data_root + 'severstal_images_divide/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        # ann_file=data_root + 'annotations/val_iron.json',
        # img_prefix=data_root + 'images_divide/val/',
        # ann_file=[data_root + 'annotations/val_iron.json',data_root + 'annotations/test_iron.json'],
        # img_prefix=[data_root + 'images_divide/val/',data_root + 'images_divide/test/'],

        ann_file=data_root + 'annotations/valAndTest_iron.json',
# ann_file=data_root + 'testNew_iron.json',
        img_prefix=data_root + 'images_divide/valAndTest/',
# img_prefix=data_root + 'testBright/',
# img_prefix=data_root + 'testClahe/',

# ann_file=data_root + 'test_iron.json',
# img_prefix=data_root + 'test/',

# ann_file=data_root + 'test_iron.json',
#         img_prefix=data_root + 'testImg/',
        # ann_file=data_root + 'severstal_annotations/val_steel.json',
        # img_prefix=data_root + 'severstal_images_divide/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',

        # ann_file=data_root + 'annotations/test_iron.json',
        # img_prefix=data_root + 'images_divide/test/',

        ann_file=data_root + 'annotations/valAndTest_iron.json',
# ann_file=data_root + 'testNew_iron.json',
        img_prefix=data_root + 'images_divide/valAndTest/',
# img_prefix=data_root + 'testBright/',
# img_prefix=data_root + 'testClahe/',

# ann_file=data_root + 'test_iron.json',
# img_prefix=data_root + 'test/',

# ann_file=data_root + 'test_iron.json',
#         img_prefix=data_root + 'testImg/',

        # ann_file=data_root + 'severstal_annotations/test_steel.json',
        # img_prefix=data_root + 'severstal_images_divide/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
