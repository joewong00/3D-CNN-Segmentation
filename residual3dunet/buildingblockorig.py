import torch.nn as nn

class SingleConv(nn.Module):
    # Group Norm + Convolution + ReLU 

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, padding=1, activation=True):
        super(SingleConv, self).__init__()

        # use only one group if the given number of groups is greater than the number of channels
        if in_channels < num_groups:
            num_groups = 1

        if activation:
            self.singleconv = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True)
            )
        
        else:
            self.singleconv = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
            )

    def forward(self, x):
        return self.singleconv(x)



class Encode(nn.Module):
    # Max-scaling then SingleConv x3 + residual 

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, pool_kernel_size=(1,2,2), num_groups=8, pooling=True, **kwargs):
        super(Encode, self).__init__()

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel_size)

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups)

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups)

        # third convolution
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups, activation=False)

        self.non_linearity = nn.ReLU(inplace=True)


    def forward(self, x):
        
        if self.pooling:
            x = self.maxpool(x)

        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out



class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, num_groups=8, scale_factor=(1, 2, 2),padding=1, **kwargs):
        super(Decode, self).__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=scale_factor, padding=padding)
        
        in_channels = out_channels
        
         # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups)

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups)

        # third convolution
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=conv_kernel_size, num_groups=num_groups, activation=False)

        self.non_linearity = nn.ReLU(inplace=True)
        

    def forward(self, encoder_features, x):

        # get the spatial dimensions of the output given the encoder_features
        output = encoder_features.size()[2:]
        x = self.upsample(x, output)

        # Summation joining instead of concatenate
        x = encoder_features + x
  
        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out
