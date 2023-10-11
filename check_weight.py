import torch

path = 'segmentation/pretrained/denseclip_fpn_vit-b.pth'
state_dict = torch.load(path)
state_dict = state_dict['state_dict']
for k in state_dict.keys():
    print(k)

print(' ')

path = 'segmentation/pretrained/VisionEncoder.pt'
state_dict = torch.load(path)
for k in state_dict.keys():
    print(k)