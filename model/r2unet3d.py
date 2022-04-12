import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import number_of_features_per_level

"""Reference
Recurrent Residual Unet 3D implemented based on https://arxiv.org/pdf/2105.02290.pdf.
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



class RRCU(nn.Module):
    # Recurrent Residual Convolutional Unit

    def __init__(self, out_channels, t=2, kernel_size=3, **kwargs):
        super(RRCU, self).__init__()

        self.t = t

        self.conv = SingleConv(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self,x):

        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)

        return x1



class RRConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,t=2, kernel_size=3):
        super(RRConvBlock,self).__init__()
        
        self.module = RRCU(out_channels=out_channels,t=t, kernel_size=kernel_size)

        self.Conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self,x):

        x = self.Conv_1x1(x)
        x1 = self.module(x)

        return x+x1



class Encode(nn.Module):
    def __init__(self, in_channels, out_channels ,t=2, conv_kernel_size=3, pool_kernel_size=(1,2,2), pooling=True):
        super(Encode,self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.module = RRConvBlock(in_channels=in_channels, out_channels=out_channels, t=t, kernel_size=conv_kernel_size)

    def forward(self,x):

        if self.pooling:
            x = self.maxpool(x)

        x = self.module(x)

        return x



class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(1, 2, 2),padding=1, mode="nearest", **kwargs):
        super(Decode, self).__init__()

        self.upsample = InterpolateUpsampling(mode)
        self.conv = SingleConv(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding)
        self.module = RRConvBlock(in_channels=in_channels, out_channels=out_channels, t=2, kernel_size=conv_kernel_size)

    
    def forward(self, encoder_features, x):

        # get the spatial dimensions of the output given the encoder_features
        output = encoder_features.size()[2:]
        x = self.upsample(x, output)

        x = self.conv(x)

        # concatenate joining
        x = torch.cat((encoder_features, x), dim=1)

        x = self.module(x)
        return x



class R2UNet3D(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512],
                testing=False,
                **kwargs):
        super(R2UNet3D, self).__init__()

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
        
        self.up1 = Decode(f_maps[3], f_maps[2])
        self.up2 = Decode(f_maps[2], f_maps[1])
        self.up3 = Decode(f_maps[1], f_maps[0])
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        print(x.shape)
        x1 = self.down1(x)
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x4 = self.down4(x3)
        print(x4.shape)

        x = self.up1(x3, x4)
        print(x.shape)
        x = self.up2(x2, x)
        print(x.shape)
        x = self.up3(x1, x)
        print(x.shape)

        y = self.final_conv(x)
        print(y.shape)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y

