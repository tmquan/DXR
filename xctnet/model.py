import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import repeat

import math

"""
From Official Pytorch Code of ResNet
        - Github : 
        - Paper : 
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


"""
    Official Code of CBAM : Convolutional Block Attention Module (ECCV 2018)
        - Github : https://github.com/Jongchan/attention-module/tree/master
        - Paper : https://arxiv.org/abs/1807.06521v2
"""
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAMModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAMModule, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
            
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    

"""
    Official Code of ECA-Net : Efficient Channel Attention (CVPR 2020)
        - Github : https://github.com/BangguWu/ECANet/tree/master
        - Paper : https://arxiv.org/abs/1910.03151
"""
class ECAModule(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


"""
New Inception Module defined in XctNet
"""
class NewInceptionModuleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, channel_3x3x3, channel_5x5x5):
        super(NewInceptionModuleBlock, self).__init__()
        self.layer0 = nn.Sequential(
            nn.ConvTranspose3d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
        self.deConv3x3x3 = nn.Sequential(
            nn.ConvTranspose3d(out_channel, channel_3x3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(channel_3x3x3),
            nn.ReLU()
        )
        
        self.deConv5x5x5 = nn.Sequential(
            nn.ConvTranspose3d(out_channel, channel_5x5x5, kernel_size=5, padding=2),
            nn.BatchNorm3d(channel_5x5x5),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.layer0(x)
        deConv3 = self.deConv3x3x3(out)
        deConv5 = self.deConv5x5x5(out)
        
        out = torch.cat((deConv3, deConv5), 1) 
        
        return out
            
class MultiScaleFeatureFusionModule(nn.Module):
    def __init__(self, depth, channel):
        super(MultiScaleFeatureFusionModule, self).__init__()
        self.transformation_module = nn.Sequential(
            nn.ConvTranspose3d(channel, channel, kernel_size=(depth, 1, 1), stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(channel)
        )
        self.make_low_feature_map = nn.AvgPool3d(2)
        
        self.w = nn.ConvTranspose3d(channel, channel, kernel_size=(2, 2, 2), stride=2, padding=0, bias=False)
        
    def forward(self, x):
        x = repeat(x, "b c h w -> b c d h w", d=1)
        out_h = self.transformation_module(x)
        out_l = self.make_low_feature_map(out_h)
        upscaled_l = self.w(out_l)
        
        out = out_h + upscaled_l
        return out

class XctNet(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super(XctNet, self).__init__()
        self.inplanes = 64
        self.res_block1 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.res_block2 = self._make_layer(block, 64, layers[0], stride = 1)
        self.res_block3 = self._make_layer(block, 128, layers[1], stride = 2)
        self.res_block4 = self._make_layer(block, 256, layers[2], stride = 2)
        self.res_block5 = self._make_layer(block, 512, layers[3], stride = 2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.cbam_module1 = CBAMModule(gate_channels=64)
        self.cbam_module2 = CBAMModule(gate_channels=512)
        
        self.eca_module1 = ECAModule(channel=64)
        self.eca_module2 = ECAModule(channel=128)
        self.eca_module3 = ECAModule(channel=256)
        
        self.up_layer1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up_layer2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up_layer3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up_layer4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up_layer5 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        
        self.inception_module1 = NewInceptionModuleBlock(256, 80, 156, 100)
        self.inception_module2 = NewInceptionModuleBlock(128, 50, 68, 60)
        self.inception_module3 = NewInceptionModuleBlock(64, 20, 34, 30)
        self.inception_module4 = NewInceptionModuleBlock(32, 10, 18, 14)
        
        self.trans_layer1 = MultiScaleFeatureFusionModule(32, 64)
        self.trans_layer2 = MultiScaleFeatureFusionModule(16, 128)
        self.trans_layer3 = MultiScaleFeatureFusionModule(8, 256)
        self.trans_layer4 = MultiScaleFeatureFusionModule(4, 512)
        
        self.final_layer = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        down1 = self.res_block1(x)
        down1 = self.maxpool(down1)
        down1 = self.cbam_module1(down1)  # [4, 64, 32, 32]
        
        down1 = self.res_block2(down1)
        down1 = self.eca_module1(down1)  # [4, 64, 32, 32]
        
        down2 = self.res_block3(down1)
        down2 = self.eca_module2(down2)  # [4, 128, 16, 16]
        
        down3 = self.res_block4(down2)
        down3 = self.eca_module3(down3)  # [4, 256, 8, 8]
        
        down4 = self.res_block5(down3)
        down4 = self.cbam_module2(down4)  # [4, 512, 4, 4]
        
        trans_down1 = self.trans_layer1(down1)
        trans_down2 = self.trans_layer2(down2)
        trans_down3 = self.trans_layer3(down3)
        trans_down4 = self.trans_layer4(down4)
        
        up1 = self.up_layer1(trans_down4)
        up1 = self.inception_module1(up1)  # [4, 256, 8, 8, 8]
        
        up2 = up1 + trans_down3
        up2 = self.up_layer2(up2)
        up2 = self.inception_module2(up2)  # [4, 128, 16, 16, 16]
        
        up3 = up2 + trans_down2
        up3 = self.up_layer3(up3)
        up3 = self.inception_module3(up3)  # [4, 64, 32, 32, 32]

        up4 = up3 + trans_down1
        up4 = self.up_layer4(up4)
        up4 = self.inception_module4(up4)  # [4, 32, 64, 64, 64]
        
        out = self.up_layer5(up4)  # [4, 16, 128, 128, 128]
        out = self.final_layer(out)
        return out
