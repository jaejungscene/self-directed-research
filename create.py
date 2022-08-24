import torch.nn as nn
import torch
from pytorch_pretrained_vit import ViT
from torchvision import models
import resnet as RN
from distillationloss import DistillationLoss

def create_criterion(args, numclass):
    if args.distil == 1:  #  <---------------- implement it yourself before training
        print('=> distil type :',args.distil_type)
        print('=> distil :',args.distil)
        teacher = RN.ResNet(args.dataset, 50, numclass, args.insize, True)
        teacher  = nn.DataParallel(teacher).cuda()
        checkpoint = torch.load('/home/ljj0512/private/project/log/2022-08-23 12:20:26/checkpoint.pth.tar')
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.eval()
        criterion = DistillationLoss(
            nn.CrossEntropyLoss().cuda(), teacher, args.distil_type, 0.5, 1.0
        )
    elif args.distil == 2:  #  <---------------- implement it yourself before training
        print('=> distil type :',args.distil_type)
        print('=> distil :',args.distil)
        teacher = create_PT_DeiT(numclass, args.insize, 0)
        teacher  = nn.DataParallel(teacher).cuda()
        checkpoint = torch.load('/home/ljj0512/private/project/log/2022-08-23 12:02:30/checkpoint.pth.tar')
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.eval()
        criterion = DistillationLoss(
            nn.CrossEntropyLoss().cuda(), teacher, args.distil_type, 0.5, 1.0
        )
    elif args.distil == 3:  #  <---------------- implement it yourself before training
        print('=> distil type :',args.distil_type)
        print('=> distil :',args.distil)
        teacher = RN.ResNet(args.dataset, 50, numclass, args.insize, True)
        teacher  = nn.DataParallel(teacher).cuda()
        checkpoint = torch.load('/home/ljj0512/private/project/log/2022-08-23 12:20:26/checkpoint.pth.tar')
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.eval()

        teacher01 = create_PT_DeiT(numclass, args.insize, 1)
        teacher01  = nn.DataParallel(teacher01).cuda()
        # checkpoint01 = torch.load('/home/ljj0512/private/project/log/2022-08-23 12:02:30/checkpoint.pth.tar')
        checkpoint01 = torch.load('/home/ljj0512/private/project/log/2022-08-24 02:00:54/checkpoint.pth.tar')
        teacher01.load_state_dict(checkpoint01['state_dict'])
        teacher01.eval()

        criterion = DistillationLoss(
            nn.CrossEntropyLoss().cuda(), teacher, args.distil_type, 0.5, 1.0, teacher01
        )
    elif args.distil == 4:  #  <---------------- implement it yourself before training
        print('=> distil type :',args.distil_type)
        print('=> distil :',args.distil)
        teacher = RN.ResNet(args.dataset, 50, numclass, args.insize, True)
        teacher  = nn.DataParallel(teacher).cuda()
        checkpoint = torch.load('/home/ljj0512/private/project/log/2022-08-23 12:20:26/checkpoint.pth.tar')
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.eval()
        
        teacher01 = RN.ResNet(args.dataset, 50, numclass, args.insize, True)
        teacher01  = nn.DataParallel(teacher01).cuda()
        checkpoint01 = torch.load('/home/ljj0512/private/project/log/2022-08-24 02:02:44/checkpoint.pth.tar')
        teacher01.load_state_dict(checkpoint01['state_dict'])
        teacher01.eval()

        criterion = DistillationLoss(
            nn.CrossEntropyLoss().cuda(), teacher, args.distil_type, 0.5, 1.5, teacher01
        )
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    return criterion


def create_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(    model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True   )
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(  model.parameters(),
                                        args.lr,
                                        weight_decay=args.weight_decay )
    return optimizer


def create_model(args, numberofclass):
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.insize, args.bottleneck)
    elif args.net_type == 'se-resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.insize, args.bottleneck, se=True)
    elif args.net_type == 'cbam-resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.insize, args.bottleneck, cbam=True)
    elif args.net_type == 'pretrained-resnet':
        model = create_PT_resnet50(numberofclass, args.insize, args.add_classifier)
    elif args.net_type == 'pretrained-vit':
        model = create_PT_ViT(numberofclass, args.insize)
    elif args.net_type == 'pretrained-deit':
        model = create_PT_DeiT(numberofclass, args.insize, args.distil)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))
    print(model)
    return model



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



def create_PT_DeiT(numclass, insize, distil=False):
    if distil > 0:
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224', pretrained=True)
        model.head = nn.Linear(384, numclass)
        model.head_dist = nn.Linear(384, numclass)
    else:
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        model.head = nn.Linear(384, numclass)   
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


# @InProceedings{pmlr-v139-touvron21a,
#   title =     {Training data-efficient image transformers &amp; distillation through attention},
#   author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
#   booktitle = {International Conference on Machine Learning},
#   pages =     {10347--10357},
#   year =      {2021},
#   volume =    {139},
#   month =     {July}
# }

