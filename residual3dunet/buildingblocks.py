import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConv(nn.Module):
    # Group Norm + Convolution + ReLU 

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, padding=1, activation=True):
        super(SingleConv, self).__init__()

        # use only one group if the given number of groups is greater than the number of channels
        if in_channels < num_groups:
            num_groups = 1

        assert in_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={in_channels}, num_groups={num_groups}'

        if activation:
            self.singleconv = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True)
            )
        
        else:
            self.singleconv = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            )

    def forward(self, x):
        return self.singleconv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, num_groups=8, padding=1):
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
        self.conv1 = SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, num_groups, padding=padding)
        # conv2
        self.conv2 = SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, num_groups, padding=padding)

    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)
        return out


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

        self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, pool_kernel_size=(1,2,2), basic_module=DoubleConv, pooling=True, num_groups=8, padding=1):
        super(Encoder, self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.basic_module = basic_module(in_channels, out_channels, 
                                        encoder=True,  
                                        kernel_size=conv_kernel_size, 
                                        num_groups=num_groups, 
                                        padding=padding)

    def forward(self, x):
        if self.pooling:
            x = self.maxpool(x)

        out = self.basic_module(x)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(1, 2, 2), basic_module=DoubleConv, num_groups=8, padding=1, mode='nearest'):
        super(Decoder, self).__init__()

        if basic_module == DoubleConv:
            self.upsampling = InterpolateUpsampling(mode)
            self.concat = True

        else:
            self.upsampling = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=scale_factor, padding=padding)
            self.concat = False 
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels, 
                                        encoder=False,  
                                        kernel_size=conv_kernel_size, 
                                        num_groups=num_groups, 
                                        padding=padding)

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output = encoder_features.size()[2:]
        x = self.upsampling(x, output)

        if self.concat:
            # Concatenate joining
            x = torch.cat((encoder_features, x), dim=1)

        else:
            # Summation joining
            x = encoder_features + x

        out = self.basic_module(x)

        return out


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, num_groups, pool_kernel_size):

    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num, 
                            pooling=False, 
                            basic_module=basic_module, 
                            conv_kernel_size=conv_kernel_size, 
                            pool_kernel_size=pool_kernel_size, 
                            num_groups=num_groups, 
                            padding=conv_padding)

        else:
            encoder = Encoder(f_maps[i-1], out_feature_num, 
                            pooling=True, 
                            basic_module=basic_module, 
                            conv_kernel_size=conv_kernel_size, 
                            pool_kernel_size=pool_kernel_size, 
                            num_groups=num_groups, 
                            padding=conv_padding)

        encoders.append(encoder)
    
    return nn.ModuleList(encoders)
        

def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, num_groups):
    
    decoders = []
    reversed_f_maps = list(reversed(f_maps))

    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature = reversed_f_maps[i] + reversed_f_maps[i+1]
        else:
            in_feature = reversed_f_maps[i]

        out_feature = reversed_f_maps[i+1]

        decoder = Decoder(in_feature, out_feature,
                            basic_module=basic_module,
                            conv_kernel_size=conv_kernel_size,
                            num_groups=num_groups,
                            padding=conv_padding)

        decoders.append(decoder)

    return nn.ModuleList(decoders)


class InterpolateUpsampling(nn.Module):
    def __init__(self, mode):
        super(InterpolateUpsampling, self).__init__()

        self.mode = mode

    def forward(self, x, size):
        return F.interpolate(x,size=size, mode=self.mode)