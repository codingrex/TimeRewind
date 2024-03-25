#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel

from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available


from models.EMA import get_EMA

from torch.utils.data import Dataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")



import torch.nn as nn



class TestDataset(Dataset):
    def __init__(self, base_folder, num_samples=100000, width=1024, height=576, sample_frames=14):
        self.num_samples = num_samples
        self.base_folder = base_folder
        self.folders = sorted(os.listdir(self.base_folder))
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx= 0):

        folder_idx= idx[0]
        frame_idx= idx[1]
        nparts= idx[2]

        chosen_folder = self.folders[folder_idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)


        rgb_folder_path = os.path.join(folder_path, 'image')

        ev_folder_path = os.path.join(folder_path, 'event')
        
        frames = os.listdir(ev_folder_path)

        frames.sort()

        start_idx= 0

        selected_frames = frames[start_idx:start_idx + self.sample_frames]


        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(rgb_folder_path, frame_name)


            if i == 0:
                init_frame_path = frame_path
        
        


        ev_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))


        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(ev_folder_path, frame_name)

            with Image.open(frame_path) as img:

                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                img_normalized = img_tensor / 127.5 - 1


                if self.channels == 3:
                    img_normalized = img_normalized.permute(
                        2, 0, 1)  # For RGB images
                    

                ev_pixel_values[i] = img_normalized



        return {'init_frame': init_frame_path, 'start_idx': start_idx, 'ev_values': ev_pixel_values}



def export_to_gif(frames, output_gif_path):

    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=140,
                       loop=0)
    

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run inference for the demo."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="The pretrained model directory.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="The input data dir.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()

    return args


   
  

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parse_args()

    EMA_PATH= args.model_dir
 
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        EMA_PATH,
        subfolder="model_0",
        low_cpu_mem_usage=True,
    )


    EMA= get_EMA()
    for i in range(len(EMA)):
        EMA[i] = EMA[i].from_pretrained(
            EMA_PATH,
            subfolder=f"model_list_{i}",
        )
        EMA[i].requires_grad_(False)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)




    weight_dtype = torch.float16
   
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    unet.to(device, dtype=weight_dtype)

    EMA = [block.to(device, dtype=weight_dtype) for block in EMA]



    TEST_DATA_PATH= args.data_dir


    test_dataset = TestDataset(width=args.width, height=args.height, sample_frames=14, base_folder= TEST_DATA_PATH)




    # Inference!

    print("***** Running Inferences *****")



    count= 0

    num_folders= len(test_dataset.folders)

    NUM_PARTS_PER_FOLDER= 1


    pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            unet=unet,
            image_encoder=
                image_encoder,
            vae=vae,
            revision=None,
            torch_dtype=weight_dtype,
        )
    

    pipeline.enable_model_cpu_offload()

    for epoch in range(num_folders):
        for i in range(NUM_PARTS_PER_FOLDER):
            batch= test_dataset[epoch, i, NUM_PARTS_PER_FOLDER]
    
        
            val_save_dir = os.path.join(
                args.output_dir, "test_images")

            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)

            out_file = os.path.join(val_save_dir,f"step_{str(count).zfill(4)}.mp4")


            valid_batch= batch

            
            conditional_pixel_values_ev= valid_batch["ev_values"].to(weight_dtype).to(
                device, non_blocking=True
            )

            conditional_pixel_values_ev = conditional_pixel_values_ev.unsqueeze(0)


            conditional_latents_ev = tensor_to_vae_latent(conditional_pixel_values_ev, vae)
            conditional_latents_ev = conditional_latents_ev / vae.config.scaling_factor

            init_frame= valid_batch['init_frame']


            print('init_frame', init_frame)



            video_frames = pipeline(
                    load_image(init_frame).resize((args.width, args.height)),
                    height=args.height,
                    width=args.width,
                    num_frames=14,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0,
                    event_latents= conditional_latents_ev,
                    EMA= EMA
            ).frames[0]




            valid_video_folder= TEST_DATA_PATH

            video_folder= test_dataset.folders[epoch]

            valid_video_folder= os.path.join(valid_video_folder, video_folder ,'image')

            video_frames= np.array(video_frames)

            

            export_to_gif(video_frames, out_file)


            count+= 1

    


if __name__ == "__main__":
    main()
