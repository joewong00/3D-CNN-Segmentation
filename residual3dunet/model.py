import torch.nn as nn
from residual3dunet.buildingblocks import Encode, Decode

class ResidualUNet3D(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels,  
                f_maps = [32,64,128,256,512],
                testing=False,
                **kwargs):
        super(ResidualUNet3D, self).__init__()

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
        print(x1.shape)
        x2 = self.down2(x1)
        print(x2.shape)
        x3 = self.down3(x2)
        print(x3.shape)
        x4 = self.down4(x3)
        print(x4.shape)

        x5 = self.down5(x4)
        print(x5.shape)

        x = self.up1(x4, x5)
        print(x.shape)
        x = self.up2(x3, x)
        print(x.shape)
        x = self.up3(x2, x)
        print(x.shape)
        x = self.up4(x1, x)
        print(x.shape)
        y = self.final_conv(x)
        print(y.shape)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        if self.testing:
            y = self.final_activation(y)

        return y
