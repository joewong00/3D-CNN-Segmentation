import torch.nn as nn
from residual3dunet.buildingblocks import DoubleConv, ExtResNetBlock, RRCNN_block, create_decoders, create_encoders
from utils.utils import number_of_features_per_level

# Source: https://github.com/wolny/pytorch-3dunet
class ResidualUNet3D(nn.Module):
    """Reference
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """
    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512,1024],
                num_levels = 5,
                testing=False,
                conv_kernel_size=3, 
                pool_kernel_size=(1,2,2), 
                num_groups=8,
                conv_padding=1,
                **kwargs):
        super(ResidualUNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing

        self.encoders = create_encoders(in_channels,f_maps, basic_module=ExtResNetBlock, 
                                        conv_kernel_size=conv_kernel_size, 
                                        conv_padding=conv_padding, 
                                        num_groups=num_groups, 
                                        pool_kernel_size=pool_kernel_size)

        self.decoders = create_decoders(f_maps, basic_module=ExtResNetBlock,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        num_groups=num_groups)

        self.final_activation = nn.Sigmoid()
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)

            encoder_features.insert(0,x)

        encoder_features = encoder_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoder_features):
            x = decoder(encoder_features, x)

        y = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y


class UNet3D(nn.Module):
    """Reference:
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>`.
    """
    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512],
                num_levels = 4,
                testing=False,
                conv_kernel_size=3, 
                pool_kernel_size=(1,2,2), 
                num_groups=8,
                conv_padding=1,
                **kwargs):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing

        self.encoders = create_encoders(in_channels,f_maps, basic_module=DoubleConv, 
                                        conv_kernel_size=conv_kernel_size, 
                                        conv_padding=conv_padding, 
                                        num_groups=num_groups, 
                                        pool_kernel_size=pool_kernel_size)

        self.decoders = create_decoders(f_maps, basic_module=DoubleConv,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        num_groups=num_groups)

        self.final_activation = nn.Sigmoid()
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0,x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        y = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y


class R2UNet3D(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [64,128,256,512],
                num_levels = 4,
                testing=False,
                conv_kernel_size=3, 
                pool_kernel_size=(1,2,2), 
                num_groups=8,
                conv_padding=1,
                **kwargs):
        super(R2UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.testing = testing

        self.encoders = create_encoders(in_channels,f_maps, basic_module=DoubleConv, 
                                        conv_kernel_size=conv_kernel_size, 
                                        conv_padding=conv_padding, 
                                        num_groups=num_groups, 
                                        pool_kernel_size=pool_kernel_size)

        self.decoders = create_decoders(f_maps, basic_module=DoubleConv,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        num_groups=num_groups)

        self.final_activation = nn.Sigmoid()
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0,x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        y = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y