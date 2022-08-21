import torch.nn as nn
from pytorch_pretrained_vit import ViT
from torchvision import models

def create_PT_ViT(numclass, insize, num_model=0):
    model_name = {0:'B_16', 1:'B_32', 2:'L_16', 3:'L_32'}
    model = ViT(    
                    model_name[num_model],
                    pretrained=True,
                    num_classes=numclass
                )
    # if insize == 32:
    #     model.patch_embedding = nn.Conv2d(3, 7)
    
    return model


def create_PT_resnet50(numclass, insize, add_classifier=False):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if add_classifier == True:
        model.fc = nn.Sequential(
            model.fc,
            nn.Linear(1000, numclass)
        )
    else:
        model.fc = nn.Linear(2048, numclass)

    # if insize == 32:
    return model



# class my_vit(nn.Module):
#     def __init__(self, sub_model) -> None:
#         super(self, my_vit).__init__()
#         preprocess = 
#             sub_model
