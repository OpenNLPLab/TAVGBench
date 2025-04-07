import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch
import numpy as np
from collections import OrderedDict
from omegaconf import OmegaConf
from tavdiffusion.models.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tavdiffusion.utils.util import save_videos_grid, save_audio
from tavdiffusion.pipelines.pipeline_video import TAVGPipeline_video
from tavdiffusion.pipelines.pipeline_audio import TAVGPipeline_audio
from tavdiffusion.models.audio_models import build_audio_pretrained_models


def define_diffusion_modules(pretrained_model_path, config):
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs)
    )
    return vae, tokenizer, text_encoder, noise_scheduler, unet

def load_pretrained_unet(unet, unet_checkpoint_path):
    unet_checkpoint = torch.load(unet_checkpoint_path, map_location="cpu")
    state_dict = unet_checkpoint["state_dict"] if "state_dict" in unet_checkpoint else unet_checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    m, u = unet.load_state_dict(new_state_dict, strict=True)

    return unet

def main():
    config = OmegaConf.load("./config.yaml")
    pretrained_model_path = config.pretrained_model_path
    unet_pretrained_path = config.unet_pretrained_path
    save_results_dir = config.save_results_dir
    global_seed = config.seed
    prompt = config.prompt
    guidance_scale = config.guidance_scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_results_dir, exist_ok=True)

    vae, tokenizer, text_encoder, noise_scheduler, unet = define_diffusion_modules(pretrained_model_path, config)
    audio_vae = build_audio_pretrained_models("audioldm-s-full").to(device)
    unet = load_pretrained_unet(unet, unet_pretrained_path)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.eval()

    tavgpipeline_video = TAVGPipeline_video(
        unet=unet, 
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler
    ).to(device)

    tavgpipeline_audio = TAVGPipeline_audio(
        audio_unet=unet.audio_unet, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler
    ).to(device)
    
    tavgpipeline_video.enable_xformers_memory_efficient_attention()
    tavgpipeline_video.enable_vae_slicing()

    generator = torch.Generator(device=device)
    generator.manual_seed(global_seed)

    video_sample = tavgpipeline_video(
        prompt,
        generator=generator,
        video_length=20,
        height=256,
        width=320,
        num_inference_steps=40,
        guidance_scale=guidance_scale,
    ).videos

    audio_latents = tavgpipeline_audio(
        prompt,
        generator=generator,
        num_inference_steps=40,
        guidance_scale=guidance_scale,
    ).audios

    audio_sample = audio_vae.decode_to_waveform(audio_vae.decode_first_stage(audio_latents))
    save_videos_grid(video_sample, f"{save_results_dir}/{prompt[0:20]}-{global_seed}.gif")
    save_audio(np.ravel(audio_sample), f"{save_results_dir}/{prompt[0:20]}-{global_seed}.wav")


if __name__ == "__main__":
    main()
