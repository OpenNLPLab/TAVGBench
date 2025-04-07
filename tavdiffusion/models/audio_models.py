import torch
from tavdiffusion.utils.audio_vae_utils import get_metadata, default_audioldm_config
from audioldm.variational_autoencoder import AutoencoderKL as AudioAutoencoderKL


def build_audio_pretrained_models(name):
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    audio_vae = AudioAutoencoderKL(**vae_config)
    audio_vae.load_state_dict(vae_state_dict)
    audio_vae.eval()

    return audio_vae