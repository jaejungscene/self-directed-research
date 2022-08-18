from pytorch_pretrained_vit import ViT
model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)