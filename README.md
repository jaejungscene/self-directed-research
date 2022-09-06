## self-directed research
- implemented model: resnet, se-resnet, cbam-resnet
- pretrained and fine-tuned model: ViT, DeiT

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