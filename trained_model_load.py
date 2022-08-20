import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import resnet as RN

checkpoint = torch.load('/home/ljj0512/private/project/log/2022-08-20 10:15:06/checkpoint.pth.tar')
print(type(checkpoint['state_dict']))
model = RN.ResNet('cifar100', 50, 100, 224, True)
model.load_state_dict(checkpoint['state_dict'])