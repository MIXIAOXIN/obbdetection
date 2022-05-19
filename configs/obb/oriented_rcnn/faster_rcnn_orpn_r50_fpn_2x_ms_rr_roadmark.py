_base_ = './faster_rcnn_orpn_r50_fpn_2x_roadmark.py'

# dataset
dataset_type = 'ROADMARKDataset'
data_root = './../data/roadmarking-30m/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    #dict(type='LoadDOTASpecialInfo'),
    #dict(type='ClassBalancedDataset', oversample_thr=0.05), # use imgset alternatively
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.0, v_flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 90), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1024, 1024)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]

# does evaluation while training
# uncomments it  when you need evaluate every epoch
# data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=4,
    # train=dict(
        # type=dataset_type,
        # task='Task1',
        # ann_file=data_root + 'train/annfiles/',
        # img_prefix=data_root + 'train/images/',
        # pipeline=train_pipeline),
    # val=dict(
        # type=dataset_type,
        # task='Task1',
        # ann_file=data_root + 'val/annfiles/',
        # img_prefix=data_root + 'val/images/',
        # pipeline=test_pipeline),
    # test=dict(
        # type=dataset_type,
        # task='Task1',
        # ann_file=data_root + 'val/annfiles/',
        # img_prefix=data_root + 'val/images/',
        # pipeline=test_pipeline))
# evaluation = dict(metric='mAP')

# disable evluation, only need train and test
# uncomments it when use trainval as train
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,  # for run： 4
    train=dict(
        type=dataset_type,
        task='Task1',
        imgset=data_root + 'trainval-aug.txt',
        ann_file=data_root + 'labelTxt/',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        task='Task1',
        imgset=data_root + 'test.txt',
        ann_file=data_root + 'labelTxt/',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = None


