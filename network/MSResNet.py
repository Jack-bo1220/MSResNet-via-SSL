import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.functional import upsample as Up
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3: newconv1.weight.data[:, 3:in_channels, :, :].copy_(
            resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)


class MSDC(nn.Module):
    def __init__(self, channel, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(MSDC, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self.create(features, size) for size in sizes])
        self.bottle = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def create(self, features, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(pool, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        fusion = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottle(torch.cat(fusion, 1))
        x = self.relu(bottle)
        dilate1_out = nonlinearity(self.conv1x1(self.dilate1(x)))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x))))))
        dilate5_out = nonlinearity(self.conv1x1(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x)))))))
        dilate6_out = self.pooling(x)
        
        out_feature = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out
        
        return out_feature

class MKMP(nn.Module):
    def __init__(self, channels):
        super(MKMP, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.maxpool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.conv2d = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        height = x.size(2)
        width = x.size(3)
        self.subpart1 = self.conv2d(self.maxpool1(x))
        self.sub1 = Up(self.subpart1, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart2 = self.conv2d(self.maxpool2(x))
        self.sub2 = Up(self.subpart2, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart3 = self.conv2d(self.maxpool3(x))
        self.sub3 = Up(self.subpart3, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart4 = self.conv2d(self.maxpool4(x))
        self.sub4 = Up(self.subpart4, size=(height, width), mode='bilinear', align_corners=False)
        
        out_feature = torch.cat([self.sub1, self.sub2, self.sub3, self.sub4, x], 1)

        return out_feature

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MSResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(MSResNet, self).__init__()

        filters = [64, 128, 256, 516]
        self.FCN = FCN(in_channels, num_classes, pretrained=True)

        self.res1 = models.resnet34(pretrained=True).layer3
        self.res2 = models.resnet34(pretrained=True).layer4
        self.res3 = models.resnet34(pretrained=True).layer4
        for n, m in self.res3.named_modules():
            if 'conv1' in n or 'downsample.0' in n: m.stride = (1, 1)

        self.decoder6 = DecoderBlock(516, 512)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = nn.Sequential(conv1x1(512, 256),nn.BatchNorm2d(256), nn.ReLU())
        self.decoder3 = nn.Sequential(conv1x1(256, 128),nn.BatchNorm2d(128), nn.ReLU())
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, filters[0])

        self.MSDC = MSDC(512,512)
        self.MKMP = MKMP(512)

        self.classifier_aux = nn.Sequential(conv1x1(512, 128), nn.BatchNorm2d(128), nn.ReLU(),
                                            conv1x1(128, num_classes, bias=True))

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x_size = x.size()
        x = self.FCN.layer0(x)  
        x = self.FCN.maxpool(x)  
        x = self.FCN.layer1(x)  
        e2 = self.FCN.layer2(x)  
        e3 = self.res1(e2)  
        e4 = self.res2(e3)  

        #center
        e4 = self.MSDC(e4)
        e4 = self.MKMP(e4)

        e3 = self.res3(e3)
        e2 = self.FCN.layer3(e2)
        e2 = self.FCN.layer4(e2)

        #Decoder
        d4 = self.decoder6(e4) + e3
        d3 = self.decoder5(d4) + e2
        d3 = self.decoder4(d3)
        d3 = self.decoder3(d3)
        d2 = self.decoder2(d3)+ x
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

if __name__ == '__main__':
    device = torch.device('cuda')
    img = torch.rand(1, 3, 448, 448)  
    net = MSResNet()
    net = net.to(device)
    img = img.to(device)
    output = net(img)
    print(output.shape)
