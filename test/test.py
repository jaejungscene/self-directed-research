import torch
import torchio as tio

x = torch.randn((1,241, 241, 165))
transforms = tio.Compose([
    tio.ToCanonical(),  # to RAS
    tio.Resample((1, 1, 1)),  # to 1 mm iso
    tio.Resize((200,200,200)),
])

# print(transforms(x).shape)
# print(tio.Resize((500,500,500))(x).shape)

import torchio as tio
from subprocess import call
root_dir = '/home/ljj0512/private/project/dataset/RSNAMICCAI'
transforms = tio.Compose([
    # tio.Resize((300,300,300)),
    tio.ToCanonical(),  # to RAS
    # tio.Resample((1, 1, 1)),  # to 1 mm iso
    tio.ZNormalization(),
])
train_set = tio.datasets.RSNAMICCAI(root_dir, train=True, transform=transforms)
test_set = tio.datasets.RSNAMICCAI(root_dir, train=False, transform=transforms)

for i in range(len(train_set)):
    print(train_set[i].T2w.data.shape)
    if(i >= 5):
        break