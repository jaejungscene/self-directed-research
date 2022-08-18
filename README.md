## self-directed research
- implemented model: resnet, se-resnet, cbam-resnet
- pretrained and fine-tuned model: ViT, MLP-Mixer

### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)

### Train Examples
- CIFAR-10, CIFAR-100: training input size = 32x32
```
python train.py --net_type resnet --dataset cifar10 --depth 50 --insize 32 --batch_size 64 --lr 0.1 --epochs 100 --no-verbose --cuda 0,1
```

- CIFAR-10, CIFAR-100: training input size = 224x224
```
python train.py --net_type resnet --dataset cifar10 --depth 50 --insize 224 --batch_size 64 --lr 0.1 --epochs 100 --no-verbose --cuda 0,1
```

- ImageNet: We used 4 GPUs to train ImageNet. 
```
python train.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 64 --lr 0.1 --epochs 100 --no-verbose --cuda 0,1
```

<h2 id="experiments">Experimental Results and Pretrained Models</h2>

- PyramidNet-200 pretrained on CIFAR-100 dataset:

Method | Top-1 Error | Model file
-- | -- | --
PyramidNet-200 [[CVPR'17](https://arxiv.org/abs/1610.02915)] (baseline) | 16.45 | [model](https://www.dropbox.com/sh/6rfew3lr761jq6c/AADrdQOXNx5tWmgOSnAw9NEVa?dl=0)
PyramidNet-200 + **CutMix** | **14.23** | [model](https://www.dropbox.com/sh/o68qbvayptt2rz5/AACy3o779BxoRqw6_GQf_QFQa?dl=0)
PyramidNet-200 + Shakedrop [[arXiv'18](https://arxiv.org/abs/1802.02375)] + **CutMix**  | **13.81** | -
PyramidNet-200 + Mixup [[ICLR'18](https://arxiv.org/abs/1710.09412)] | 15.63 | [model](https://www.dropbox.com/sh/g55jnsv62v0n59s/AAC9LPg-LjlnBn4ttKs6vr7Ka?dl=0)
PyramidNet-200 + Manifold Mixup [[ICML'19](https://arxiv.org/abs/1806.05236)] | 16.14 | [model](https://www.dropbox.com/sh/nngw7hhk1e8msbr/AABkdCsP0ABnQJDBX7LQVj4la?dl=0)
PyramidNet-200 + Cutout [[arXiv'17](https://arxiv.org/abs/1708.04552)] | 16.53 | [model](https://www.dropbox.com/sh/ajjz4q8c8t6qva9/AAAeBGb2Q4TnJMW0JAzeVSpfa?dl=0)
PyramidNet-200 + DropBlock [[NeurIPS'18](https://arxiv.org/abs/1810.12890)] | 15.73 | [model](https://www.dropbox.com/sh/vefjo960gyrsx2i/AACYA5wOJ_yroNjIjdsN1Dz2a?dl=0)
PyramidNet-200 + Cutout + Labelsmoothing | 15.61 | [model](https://www.dropbox.com/sh/1mur0kjcfxdn7jn/AADmghqrj0dXAG0qY1v3Csb6a?dl=0)
PyramidNet-200 + DropBlock + Labelsmoothing | 15.16 | [model](https://www.dropbox.com/sh/n1dn6ggyxjcoogc/AADpSSNzvaraSCqWtHBE0qMca?dl=0)
PyramidNet-200 + Cutout + Mixup | 15.46 | [model](https://www.dropbox.com/sh/5run1sx8oy0v9oi/AACiR_wEBQVp2HMZFx6lGl3ka?dl=0)


- ResNet models pretrained on ImageNet dataset:

Method | Top-1 Error | Model file
-- | -- | --
ResNet-50 [[CVPR'16](https://arxiv.org/abs/1512.03385)] (baseline) | 23.68 | [model](https://www.dropbox.com/sh/phwbbrtadrclpnx/AAA9QUW9G_xvBdI-mDiIzP_Ha?dl=0)
ResNet-50 + **CutMix** | **21.40** | [model](https://www.dropbox.com/sh/w8dvfgdc3eirivf/AABnGcTO9wao9xVGWwqsXRala?dl=0)
ResNet-50 + **Feature CutMix** | **21.80** | [model](https://www.dropbox.com/sh/zj1wptsg0hwqf0k/AABRNzvjFmIS7_vOEQkqb6T4a?dl=0)
ResNet-50 + Mixup [[ICLR'18](https://arxiv.org/abs/1710.09412)] | 22.58 | [model](https://www.dropbox.com/sh/g64c8bda61n12if/AACyaTZnku_Sgibc9UvOSblNa?dl=0)
ResNet-50 + Manifold Mixup [[ICML'19](https://arxiv.org/abs/1806.05236)] | 22.50 | [model](https://www.dropbox.com/sh/bjardjje11pti0g/AABFGW0gNrNE8o8TqUf4-SYSa?dl=0)
ResNet-50 + Cutout [[arXiv'17](https://arxiv.org/abs/1708.04552)] | 22.93 | [model](https://www.dropbox.com/sh/ln8zk2z7zt2h1en/AAA7z8xTBlzz7Ofbd5L7oTnTa?dl=0)
ResNet-50 + AutoAugment [[CVPR'19](https://arxiv.org/abs/1805.09501)] | 22.40* | -
ResNet-50 + DropBlock [[NeurIPS'18](https://arxiv.org/abs/1810.12890)] | 21.87* | -
ResNet-101 + **CutMix** | **20.17** | [model](https://www.dropbox.com/sh/1z4xnp9nwdmpzb5/AACQX4KU8XkTN0JSTfjkCktNa?dl=0)
ResNet-152 + **CutMix** | **19.20** | [model](https://www.dropbox.com/s/6vq1mzy27z8qxko/resnet152_cutmix_acc_80_80.pth?dl=0)
ResNeXt-101 (32x4d) + **CutMix** | **19.47** | [model](https://www.dropbox.com/s/maysvgopsi17qi0/resnext_cutmix.pth.tar?dl=0)

\* denotes results reported in the original papers

## Transfer Learning Results

Backbone | ImageNet Cls (%) | ImageNet Loc (%) | CUB200 Loc (%) | Detection (SSD) (mAP) | Detection (Faster-RCNN) (mAP) | Image Captioning (BLEU-4)
-- | -- | -- | -- | -- | -- | --
ResNet50 | 23.68 | 46.3 | 49.41 | 76.7 | 75.6 | 22.9
ResNet50+Mixup | 22.58 | 45.84 | 49.3 | 76.6 | 73.9 | 23.2
ResNet50+Cutout | 22.93 | 46.69 | 52.78 | 76.8 | 75 | 24.0
ResNet50+**CutMix** | **21.60** | **46.25** | **54.81** | **77.6** | **76.7** | **24.9**
