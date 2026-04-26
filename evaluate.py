"""
SSIM & FID Evaluation Script for Gen AI Assignment 1
================================================
Computes Structural Similarity Index (SSIM) and 
Frechet Inception Distance (FID) between generated
and real images for VAE, GAN, and Diffusion models.

Usage:
    python evaluate.py                    # evaluate all models
    python evaluate.py --model vae        # evaluate only VAE
    python evaluate.py --model gan        # evaluate only GAN
    python evaluate.py --model diffusion  # evaluate only Diffusion
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset


# ─────────────────────────────────────────────
#  Pure-PyTorch SSIM Implementation
#  (no skimage dependency needed)
# ─────────────────────────────────────────────

def _gaussian_kernel_1d(size, sigma):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_kernel_2d(size=11, sigma=1.5, channels=3):
    """Create a 2D Gaussian kernel for SSIM."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)     # outer product
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Compute SSIM between two image tensors.

    Args:
        img1, img2: (B, C, H, W) tensors in [0, 1] range
        window_size: Gaussian window size
        sigma: Gaussian sigma

    Returns:
        Mean SSIM value (scalar)
    """
    C = img1.size(1)
    kernel = _gaussian_kernel_2d(window_size, sigma, C).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def batch_ssim(generated, real, batch_size=32):
    """
    Compute mean SSIM between a set of generated images and
    randomly paired real images.

    Both inputs are lists/tensors of (C, H, W) images in [-1, 1] range.
    They are denormalised to [0, 1] before computing SSIM.
    """
    # Denormalise [-1, 1] → [0, 1]
    gen = (generated * 0.5 + 0.5).clamp(0, 1)
    real = (real * 0.5 + 0.5).clamp(0, 1)

    # Pair up: take min of both sets
    n = min(gen.size(0), real.size(0))
    gen  = gen[:n]
    real = real[:n]

    # Compute SSIM in batches
    ssim_scores = []
    for i in range(0, n, batch_size):
        g_batch = gen[i:i+batch_size]
        r_batch = real[i:i+batch_size]
        ssim_val = compute_ssim(g_batch, r_batch)
        ssim_scores.append(ssim_val)

    return np.mean(ssim_scores)


def batch_fid(generated, real, device, batch_size=50):
    """
    Compute Frechet Inception Distance (FID) between generated 
    and real images using torchmetrics.
    
    Both inputs are (B, C, H, W) in [-1, 1] range.
    Images are processed in small batches to avoid OOM — InceptionV3
    internally upscales to 299x299, which explodes memory at large N.
    """
    import gc
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError("Please install torchmetrics to use FID: pip install torchmetrics")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # Feed real images in small batches
    for i in range(0, real.size(0), batch_size):
        chunk = ((real[i:i+batch_size] * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        fid_metric.update(chunk.to(device), real=True)
        del chunk

    # Feed generated images in small batches
    for i in range(0, generated.size(0), batch_size):
        chunk = ((generated[i:i+batch_size] * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        fid_metric.update(chunk.to(device), real=False)
        del chunk

    score = fid_metric.compute().item()

    # Cleanup: free InceptionV3 and intermediate state
    del fid_metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return score


# ─────────────────────────────────────────────
#  Model Loaders
# ─────────────────────────────────────────────

def load_vae(checkpoint_path, device):
    """Load trained CVAE and return (model, generate_fn)."""
    from vae import CVAE

    model = CVAE(
        latent_dim=128,
        base_filters=32,
        conv_kernel=3,
        activation='relu',
        use_interpolation=False,
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    def generate(num_samples, label):
        return model.generate(num_samples, label, device)

    return model, generate


def load_gan(checkpoint_path, device):
    """Load trained unconditional GAN generator and return (model, generate_fn)."""
    from gan import Generator

    model = Generator(
        latent_dim=128,
        base_filters=64,
        activation='relu',
        dropout=0.3,
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    def generate(num_samples, label):
        # Unconditional GAN — ignores class label, generates random faces
        with torch.no_grad():
            z = torch.randn(num_samples, 128, device=device)
            return model(z)

    return model, generate


def load_diffusion(checkpoint_path, device):
    """Load trained DDPM and return (model, generate_fn)."""
    from diffusion import build_unet, GaussianDiffusion

    model = build_unet(
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=2,
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    def generate(num_samples, label):
        return diffusion.generate(model, num_samples, label, device)

    return model, generate


def load_ddgm(checkpoint_path, device):
    """Load trained DDGM generator and return (model, generate_fn)."""
    from gan import Generator

    model = Generator(
        latent_dim=128,
        base_filters=64,
        activation='relu',
        dropout=0.3,
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    def generate(num_samples, label):
        # DDGM generator is unconditional — ignores class label
        with torch.no_grad():
            z = torch.randn(num_samples, 128, device=device)
            return model(z)

    return model, generate


# ─────────────────────────────────────────────
#  Evaluation Pipeline
# ─────────────────────────────────────────────

def get_real_images(dataset, label, num_images, device):
    """
    Extract real images of a given class from the dataset.
    """
    images = []
    for i in range(len(dataset)):
        img, lbl = dataset[i]
        if lbl == label:
            images.append(img)
        if len(images) >= num_images:
            break
    return torch.stack(images).to(device)


def evaluate_model(model_name, generate_fn, dataset, device,
                   num_samples=100):
    """
    Evaluate a model by generating images and computing SSIM
    against real images for each class.
    """
    results = {}

    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        print(f"  Generating {num_samples} {label_name} images...")
        generated = generate_fn(num_samples, label).to(device)

        print(f"  Loading {num_samples} real {label_name} images...")
        real = get_real_images(dataset, label, num_samples, device)

        # Shuffle real images for random pairing
        perm = torch.randperm(real.size(0))
        real = real[perm]

        ssim_val = batch_ssim(generated, real)
        fid_val = batch_fid(generated, real, device)
        results[f'SSIM_{label_name}'] = ssim_val
        results[f'FID_{label_name}'] = fid_val
        print(f"  SSIM ({label_name}): {ssim_val:.4f}  |  FID ({label_name}): {fid_val:.4f}")

    # Overall average
    results['SSIM_Average'] = np.mean([results['SSIM_Glasses'], results['SSIM_No Glasses']])
    results['FID_Average'] = np.mean([results['FID_Glasses'], results['FID_No Glasses']])
    print(f"  SSIM (Average): {results['SSIM_Average']:.4f}  |  FID (Average): {results['FID_Average']:.4f}")

    return results


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='SSIM evaluation for Gen AI Assignment 1')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vae', 'gan', 'diffusion', 'ddgm', 'all'],
                        help='Which model to evaluate')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of images to generate per class')
    parser.add_argument('--csv_path', type=str, default='final_train.csv')
    parser.add_argument('--images_dir', type=str, default='images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset (no augmentations for evaluation)
    dataset = GlassesDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split='all',
        target_size=(64, 64),
        augment=False,
    )

    # ── Model configurations ─────────────────────────────────
    model_configs = {
        'vae': {
            'loader': load_vae,
            'checkpoint': 'vae_best.pth',
            'name': 'CVAE',
        },
        'gan': {
            'loader': load_gan,
            'checkpoint': 'gan_unc_generator_best.pth',
            'name': 'GAN',
        },
        'diffusion': {
            'loader': load_diffusion,
            'checkpoint': 'diffusion_best.pth',
            'name': 'DDPM',
        },
        'ddgm': {
            'loader': load_ddgm,
            'checkpoint': 'ddgm_generator_best.pth',
            'name': 'DDGM',
        },
    }

    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = ['vae', 'gan', 'diffusion', 'ddgm']
    else:
        models_to_eval = [args.model]

    # ── Run evaluations ──────────────────────────────────────
    all_results = {}

    for model_key in models_to_eval:
        config = model_configs[model_key]
        checkpoint = config['checkpoint']

        if not os.path.exists(checkpoint):
            print(f"\n⚠️  Skipping {config['name']}: "
                  f"checkpoint '{checkpoint}' not found")
            continue

        print(f"\n{'─' * 50}")
        print(f"Evaluating {config['name']} ({checkpoint})")
        print(f"{'─' * 50}")

        model, generate_fn = config['loader'](checkpoint, device)
        results = evaluate_model(
            config['name'], generate_fn, dataset, device,
            num_samples=args.num_samples,
        )
        all_results[config['name']] = results

    # ── Print summary table ──────────────────────────────────
    if all_results:
        print(f"\n{'═' * 95}")
        print(f"  SSIM & FID Comparison Summary")
        print(f"{'═' * 95}")
        print(f"  {'Model':<12} | {'SSIM (G)':>10} {'SSIM (NG)':>12} {'SSIM (Avg)':>10} | {'FID (G)':>10} {'FID (NG)':>12} {'FID (Avg)':>10}")
        print(f"  {'─' * 90}")
        for name, res in all_results.items():
            s_g  = res.get('SSIM_Glasses', 0)
            s_ng = res.get('SSIM_No Glasses', 0)
            s_avg = res.get('SSIM_Average', 0)
            f_g  = res.get('FID_Glasses', 0)
            f_ng = res.get('FID_No Glasses', 0)
            f_avg = res.get('FID_Average', 0)
            print(f"  {name:<12} | {s_g:>10.4f} {s_ng:>12.4f} {s_avg:>10.4f} | {f_g:>10.4f} {f_ng:>12.4f} {f_avg:>10.4f}")
        print(f"{'═' * 95}")

        # ── Save results to file ─────────────────────────────
        with open('evaluation_results.txt', 'w') as f:
            f.write("SSIM & FID Evaluation Results\n")
            f.write("=" * 95 + "\n")
            f.write(f"{'Model':<12} | {'SSIM (G)':>10} {'SSIM (NG)':>12} {'SSIM (Avg)':>10} | {'FID (G)':>10} {'FID (NG)':>12} {'FID (Avg)':>10}\n")
            f.write("-" * 90 + "\n")
            for name, res in all_results.items():
                s_g  = res.get('SSIM_Glasses', 0)
                s_ng = res.get('SSIM_No Glasses', 0)
                s_avg = res.get('SSIM_Average', 0)
                f_g  = res.get('FID_Glasses', 0)
                f_ng = res.get('FID_No Glasses', 0)
                f_avg = res.get('FID_Average', 0)
                f.write(f"{name:<12} | {s_g:>10.4f} {s_ng:>12.4f} {s_avg:>10.4f} | {f_g:>10.4f} {f_ng:>12.4f} {f_avg:>10.4f}\n")
            f.write("=" * 95 + "\n")
        print("\nResults saved to evaluation_results.txt")
