import torch.nn as nn
from utils.utils import number_of_features_per_level

"""Reference
Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf and https://arxiv.org/pdf/2006.14215.pdf.
"""

class SingleConv(nn.Module):
    # Convolution + GroupNorm + ELU

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, padding=1, activation=True):
        super(SingleConv, self).__init__()

        # use only one group if the given number of groups is greater than the number of channels
        if out_channels < num_groups:
            num_groups = 1

        assert out_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={in_channels}, num_groups={num_groups}'

        if activation:
            self.singleconv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ELU(inplace=True)
            )
        
        else:
            self.singleconv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            )

    def forward(self, x):
        return self.singleconv(x)



class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups)

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups)

        # third convolution
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups, activation=False)

        self.non_linearity = nn.ELU(inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out



class Encode(nn.Module):
    # Max-scaling then SingleConv x3 + residual 

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, pool_kernel_size=(1,2,2), num_groups=8, padding=1, pooling=True, **kwargs):
        super(Encode, self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.module = ExtResNetBlock(in_channels, out_channels, 
                                    kernel_size=conv_kernel_size, 
                                    num_groups=num_groups, 
                                    padding=padding)


    def forward(self, x):
        
        if self.pooling:
            x = self.maxpool(x)

        out = self.module(x)

        return out



class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, num_groups=8, scale_factor=(1, 2, 2),padding=1, **kwargs):
        super(Decode, self).__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=scale_factor, padding=padding)

        in_channels = out_channels
        
        self.module = ExtResNetBlock(in_channels, out_channels, 
                                    kernel_size=conv_kernel_size, 
                                    num_groups=num_groups, 
                                    padding=padding)

        

    def forward(self, encoder_features, x):

        # get the spatial dimensions of the output given the encoder_features
        output = encoder_features.size()[2:]
        x = self.upsample(x, output)

        # Summation joining instead of concatenate
        x = encoder_features + x
  
        out = self.module(x)

        return out



class ResUNet3D(nn.Module):
    
    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512,1024],
                testing=False,
                **kwargs):
        super(ResUNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=5)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing
        self.final_activation = nn.Sigmoid()

        self.down1 = Encode(in_channels, f_maps[0], pooling=False)
        self.down2 = Encode(f_maps[0], f_maps[1])
        self.down3 = Encode(f_maps[1], f_maps[2])
        self.down4 = Encode(f_maps[2], f_maps[3])
        self.down5 = Encode(f_maps[3], f_maps[4])

        self.up1 = Decode(f_maps[4], f_maps[3])
        self.up2 = Decode(f_maps[3], f_maps[2])
        self.up3 = Decode(f_maps[2], f_maps[1])
        self.up4 = Decode(f_maps[1], f_maps[0])
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        y = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y