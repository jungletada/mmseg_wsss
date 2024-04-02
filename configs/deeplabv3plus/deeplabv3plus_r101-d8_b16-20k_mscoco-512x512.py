_base_ = './deeplabv3plus_r50-d8_b16-20k_mscoco-512x512.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
