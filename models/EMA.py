from dataclasses import dataclass

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalControlnetMixin

IN_CHANNELS_UPBLOCKS= [64, 192, 384, 768, 768, 384, 192, 64]
OUT_CHANNELS_UPBLOCKS= [320, 320, 640, 1280, 1280, 1280, 1280, 640]
NUM_DOWN= [0, 1, 2, 3, 3, 2, 1, 0]


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.SiLU(inplace=True)
    )


class Conv_MH(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    _supports_gradient_checkpointing = True

    @register_to_config

    def __init__(self, in_channels, out_channels, num_down):
        super().__init__()

        self.conv_in= nn.Sequential(nn.Conv2d(4, in_channels, kernel_size=3, padding=1))

        
        self.conv_down_lits= nn.ModuleList([])

        for i in range(num_down):
            self.conv_down_lits.append(nn.Sequential(double_conv(in_channels, in_channels), nn.MaxPool2d(2)))

        
        self.conv_convert= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv_out= nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)



        self.non_linearity = nn.SiLU()
        self.time_emb_proj= nn.Linear(1280, in_channels)

    def forward(self, x, t_emb, noisy_latent= None):

        batch_size, num_frames= x.shape[0], x.shape[1]

        h, w= x.shape[3], x.shape[4]

        x= x.flatten(0, 1)

        
        x= self.conv_in(x)

        for i in range(len(self.conv_down_lits)):
            x= self.conv_down_lits[i](x)
        

        if t_emb is not None:

            unet_temb = t_emb.reshape(batch_size, num_frames, -1)

            unet_temb= self.non_linearity(unet_temb)

            unet_temb = self.time_emb_proj(unet_temb)[:, :, :, None, None]

            unet_temb= unet_temb.flatten(0, 1)

            x= x + unet_temb

        if noisy_latent is not None:

            x= self.conv_convert(x)

 
            x= torch.cat((x, noisy_latent), dim=1)

        
        return self.conv_out(x)
    

def get_conv_blocks(in_channels, out_channels, num_down):
    conv_blocks= nn.ModuleList([])
    for i in range(len(in_channels)):
        conv_blocks.append(Conv_MH(in_channels[i], out_channels[i], num_down[i]))

    return conv_blocks


def get_EMA():
    return get_conv_blocks(IN_CHANNELS_UPBLOCKS, OUT_CHANNELS_UPBLOCKS, NUM_DOWN)