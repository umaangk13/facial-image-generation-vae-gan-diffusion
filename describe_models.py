"""
Visualise all model architectures used in the project.
Uses torchinfo (pip install torchinfo) for rich summaries,
falls back to print(model) if not available.

Usage:
    python describe_models.py
"""

import os
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# ── Try torchinfo for rich summaries ────────────────────────
try:
    from torchinfo import summary as model_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("⚠️  'torchinfo' not installed. Using basic print(model).")
    print("   Install it for richer output: pip install torchinfo\n")


def describe(model, model_name, input_data=None, input_size=None):
    """Print a model's architecture."""
    print(f"\n{'═' * 70}")
    print(f"  {model_name}")
    print(f"{'═' * 70}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {param_count:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print()

    if HAS_TORCHINFO and input_data is not None:
        model_summary(model, input_data=input_data, depth=4,
                      col_names=["input_size", "output_size", "num_params", "kernel_size"],
                      verbose=1)
    elif HAS_TORCHINFO and input_size is not None:
        model_summary(model, input_size=input_size, depth=4,
                      col_names=["input_size", "output_size", "num_params", "kernel_size"],
                      verbose=1)
    else:
        print(model)

    print()


class Tee:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath, original_stdout):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = original_stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


if __name__ == '__main__':

    output_file = 'model_architectures.txt'
    tee = Tee(output_file, sys.stdout)
    sys.stdout = tee

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ─────────────────────────────────────────
    #  1. CVAE (vae.py)
    # ─────────────────────────────────────────
    from vae import CVAE

    cvae = CVAE(latent_dim=128, base_filters=32, conv_kernel=3,
                activation='relu', use_interpolation=False).to(device)

    dummy_img    = torch.randn(1, 3, 64, 64, device=device)
    dummy_label  = torch.zeros(1, dtype=torch.long, device=device)

    describe(cvae, "CVAE (Conditional VAE)",
             input_data=[dummy_img, dummy_label])

    # ─────────────────────────────────────────
    #  2. Unconditional GAN — Generator (gan.py)
    # ─────────────────────────────────────────
    from gan import Generator, Discriminator

    gen = Generator(latent_dim=128, base_filters=64,
                    activation='relu', dropout=0.3).to(device)
    disc = Discriminator(base_filters=64, dropout=0.3).to(device)

    dummy_z = torch.randn(1, 128, device=device)

    describe(gen, "GAN — Generator (Unconditional)",
             input_data=[dummy_z])

    describe(disc, "GAN — Discriminator",
             input_data=[dummy_img])

    # ─────────────────────────────────────────
    #  3. Conditional Diffusion UNet (diffusion.py)
    # ─────────────────────────────────────────
    from diffusion import build_unet

    unet = build_unet(base_dim=64, dim_mults=(1, 2, 4, 8),
                       num_res_blocks=2).to(device)

    dummy_t = torch.zeros(1, dtype=torch.long, device=device)

    describe(unet, "DDPM — Conditional UNet",
             input_data=[dummy_img, dummy_t, dummy_label])

    # ─────────────────────────────────────────
    #  Done
    # ─────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  All models described successfully!")
    print(f"  Output saved to: {output_file}")
    print(f"{'═' * 70}")

    tee.close()
    sys.stdout = tee.stdout
