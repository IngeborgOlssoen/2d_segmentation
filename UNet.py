import torch 
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def create_2d_unet():
    model=UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16,32,64,128,256),
        strides=(2,2,2,2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1
    )

    return model

unet_model=create_2d_unet()
