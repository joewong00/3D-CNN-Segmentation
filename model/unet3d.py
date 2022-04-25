import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import number_of_features_per_level

"""Reference:
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>`.
"""

class InterpolateUpsampling(nn.Module):
    def __init__(self, mode):
        super(InterpolateUpsampling, self).__init__()

        self.mode = mode

    def forward(self, x, size):
        return F.interpolate(x,size=size, mode=self.mode)


class SingleConv(nn.Module):
    # Convolution + Batch Norm + ReLU

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SingleConv, self).__init__()

        self.singleconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.singleconv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels

        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        
        # conv1
        self.conv1 = SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=padding)
        # conv2
        self.conv2 = SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=padding)


    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)
        return out


class Encode(nn.Module):
    # Max-scaling then SingleConv x3 + residual 

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, pool_kernel_size=(1,2,2), padding=1, pooling=True, **kwargs):
        super(Encode, self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.module = DoubleConv(in_channels, out_channels,
                                encoder=True, 
                                kernel_size=conv_kernel_size,  
                                padding=padding)

    def forward(self, x):
        
        if self.pooling:
            x = self.maxpool(x)

        out = self.module(x)

        return out



class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, padding=1, mode='nearest', **kwargs):
        super(Decode, self).__init__()

        self.upsample = InterpolateUpsampling(mode)
        
        self.module = DoubleConv(in_channels, out_channels, 
                                encoder=False,
                                kernel_size=conv_kernel_size,  
                                padding=padding)

        

    def forward(self, encoder_features, x):

        # get the spatial dimensions of the output given the encoder_features
        output = encoder_features.size()[2:]
        x = self.upsample(x, output)

        # concatenate joining
        x = torch.cat((encoder_features, x), dim=1)
  
        out = self.module(x)

        return out



class UNet3D(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512],
                testing=False,
                **kwargs):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=4)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing
        self.final_activation = nn.Sigmoid()

        self.down1 = Encode(in_channels, f_maps[0], pooling=False)
        self.down2 = Encode(f_maps[0], f_maps[1])
        self.down3 = Encode(f_maps[1], f_maps[2])
        self.down4 = Encode(f_maps[2], f_maps[3])
        
        self.up1 = Decode(f_maps[3] + f_maps[2], f_maps[2])
        self.up2 = Decode(f_maps[2] + f_maps[1], f_maps[1])
        self.up3 = Decode(f_maps[1] + f_maps[0], f_maps[0])
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x3, x4)
        x = self.up2(x2, x)
        x = self.up3(x1, x)

        y = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y

