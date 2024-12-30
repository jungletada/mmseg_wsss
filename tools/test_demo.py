from mmseg.models import MCTAdapter
from mmseg.models import LargeFOV
import torch


ckpt = 'work_dirs/checkpoints/mcta-deit-small-voc-7458.pth'
state_dict = torch.load(ckpt, map_location='cpu')
model_dict = state_dict['model']


print(model_dict.keys())

prefix='backbone.'
updated_model_dict = {prefix + k: v for k, v in model_dict.items()}
u_ckpt = 'work_dirs/checkpoints/mcta-deit-small-voc-pretrained.pth'
torch.save(updated_model_dict, u_ckpt)
print(updated_model_dict.keys())