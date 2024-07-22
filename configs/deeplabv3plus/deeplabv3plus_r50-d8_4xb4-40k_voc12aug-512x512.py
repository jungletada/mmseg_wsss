_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.)
        }))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))
