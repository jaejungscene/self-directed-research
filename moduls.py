import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class SEblock(nn.Module):
    r = 16
    def __init__(self, channel):
        super(SEblock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//self.r, bias=False),
            nn.ReLU(),
            nn.Linear(channel//self.r, channel, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)  # flatten
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x) # y의 형상을 x로 맞추어줌



class CBAM(nn.Module):
    r = 16
    def __init__(self, channel):
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//self.r, bias=False),
            nn.ReLU(),
            nn.Linear(channel//self.r, channel, bias=False),
        )

        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=7//2, bias=False)

    def forward(self, x):
        ## channel attention 과정 ##
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x)
        y1 = y1.view(b, c)  # flatten
        y1 = self.fc(y1)
        y1 = y1.view(b, c, 1, 1)
        y2 = self.max_pool(x)
        y2 = y2.view(b, c)  # flatten
        y2 = self.fc(y2)
        y2 = y2.view(b, c, 1, 1)
        channel_attention = nn.Sigmoid()(y1 + y2)
        x = x * channel_attention.expand_as(x) # hadamard product (y의 형상을 x로 맞추어줌)

        ## spatial attention 과정 ##
        spatial_avg = torch.mean(x, dim=1) # channel이 0으로 되어 있어서 1로 바꿈
        b, size, _, = spatial_avg.size()
        spatial_avg = spatial_avg.view(b, 1, size, -1)
        spatial_max, _ = torch.max(x, dim=1) # channel이 0으로 되어 있어서 1로 바꿈
        b, size, _, = spatial_max.size()
        spatial_max = spatial_max.view(b, 1, size, -1)
        spatial_attention = torch.cat([spatial_avg, spatial_max], 1)
        spatial_attention = self.conv7(spatial_attention)
        spatial_attention = nn.Sigmoid()(spatial_attention)
        x = x * spatial_attention.expand_as(x)
        return x