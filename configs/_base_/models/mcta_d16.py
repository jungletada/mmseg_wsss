# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes=20
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MCTAdapter',
        num_classes=num_classes),
    
    decode_head=dict(
        type='LargeFOV',
        in_channels=384,
        channels=512,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
