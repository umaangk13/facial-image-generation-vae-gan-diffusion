"""
Demo Generation Script
======================
Generates face images using trained VAE, GAN, and Diffusion model weights.
Displays and saves a grid of generated images for each model.

Usage:
    python generate_demo.py

The script will:
  1. Load each model's best checkpoint
  2. Generate sample face images
  3. Display them in a matplotlib window and save to PNG files

Models:
  - CVAE:      Conditional generation (Glasses vs No Glasses)
  - GAN:       Unconditional face generation
  - Diffusion: Conditional generation (Glasses vs No Glasses)
"""

import os
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

OUTPUT_DIR = 'demo_generated'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def denorm(tensor):
    """Convert [-1, 1] tensor to [0, 1] numpy image for display."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


# ─────────────────────────────────────────────
#  VAE Generation
# ─────────────────────────────────────────────

def generate_vae(device, num_per_class=5, weights_path='vae_best.pth'):
    from vae import CVAE

    if not os.path.exists(weights_path):
        print(f"  ⚠ VAE weights not found at {weights_path}, skipping...")
        return None

    print(f"  Loading VAE from {weights_path}...")
    model = CVAE(latent_dim=128, base_filters=32, conv_kernel=3,
                 activation='relu', use_interpolation=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    glasses_imgs    = model.generate(num_per_class, label=1, device=device)
    no_glasses_imgs = model.generate(num_per_class, label=0, device=device)

    fig, axes = plt.subplots(2, num_per_class, figsize=(3 * num_per_class, 6))
    for col in range(num_per_class):
        axes[0, col].imshow(denorm(glasses_imgs[col]))
        axes[0, col].axis('off')
        axes[1, col].imshow(denorm(no_glasses_imgs[col]))
        axes[1, col].axis('off')

    axes[0, 0].set_title('Glasses', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('No Glasses', fontsize=12, fontweight='bold')
    fig.suptitle('CVAE — Conditional Generation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'demo_vae_generated.png')
    plt.savefig(save_path, dpi=150)
    print(f"  ✅ Saved to {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────
#  GAN Generation
# ─────────────────────────────────────────────

def generate_gan(device, num_images=10, weights_path='gan_unc_generator_best.pth'):
    from gan import Generator

    if not os.path.exists(weights_path):
        print(f"  ⚠ GAN weights not found at {weights_path}, skipping...")
        return None

    print(f"  Loading GAN Generator from {weights_path}...")
    generator = Generator(latent_dim=128, base_filters=64,
                          activation='relu', dropout=0.3).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_images, 128, device=device)
        images = generator(z)

    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                axes[i, j].imshow(denorm(images[idx]))
            axes[i, j].axis('off')

    fig.suptitle('GAN — Unconditional Face Generation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'demo_gan_generated.png')
    plt.savefig(save_path, dpi=150)
    print(f"  ✅ Saved to {save_path}")
    plt.show()
    return fig



# ─────────────────────────────────────────────
#  Diffusion Generation
# ─────────────────────────────────────────────

def generate_diffusion(device, num_per_class=5, weights_path='diffusion_best.pth'):
    from diffusion import build_unet, GaussianDiffusion

    if not os.path.exists(weights_path):
        print(f"  ⚠ Diffusion weights not found at {weights_path}, skipping...")
        return None

    print(f"  Loading Diffusion model from {weights_path}...")
    print("  (This will take a minute — 1000 reverse diffusion steps per image)")

    model = build_unet(base_dim=64, dim_mults=(1, 2, 4, 8),
                       num_res_blocks=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, beta_start=1e-4,
                                   beta_end=0.02, device=device)

    print("  Generating glasses images...")
    glasses_imgs = diffusion.generate(model, num_per_class, label=1, device=device)
    print("  Generating no-glasses images...")
    no_glasses_imgs = diffusion.generate(model, num_per_class, label=0, device=device)

    fig, axes = plt.subplots(2, num_per_class, figsize=(3 * num_per_class, 6))
    for col in range(num_per_class):
        axes[0, col].imshow(denorm(glasses_imgs[col]))
        axes[0, col].axis('off')
        axes[1, col].imshow(denorm(no_glasses_imgs[col]))
        axes[1, col].axis('off')

    axes[0, 0].set_title('Glasses', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('No Glasses', fontsize=12, fontweight='bold')
    fig.suptitle('DDPM — Conditional Generation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'demo_diffusion_generated.png')
    plt.savefig(save_path, dpi=150)
    print(f"  ✅ Saved to {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("=" * 50)
    print("  1. VAE Generation")
    print("=" * 50)
    generate_vae(device)

    print("\n" + "=" * 50)
    print("  2. GAN Generation")
    print("=" * 50)
    generate_gan(device)

    print("\n" + "=" * 50)
    print("  3. Diffusion Generation")
    print("=" * 50)
    generate_diffusion(device)

    print(f"\n🎉 Demo complete! Generated images saved in {OUTPUT_DIR}/")
    print(f"   - {OUTPUT_DIR}/demo_vae_generated.png")
    print(f"   - {OUTPUT_DIR}/demo_gan_generated.png")
    print(f"   - {OUTPUT_DIR}/demo_diffusion_generated.png")
