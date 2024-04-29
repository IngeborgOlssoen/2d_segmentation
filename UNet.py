import torch 
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def create_2d_unet():
    model=UNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=2,
        channels=(2,4,8,16,32), #la til to lag
        strides=(2,2,2,2),
        num_res_units=4,
        norm=Norm.INSTANCE,
        dropout=0.01
    )

    return model

unet_model=create_2d_unet()
