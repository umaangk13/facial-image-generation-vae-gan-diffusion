"""
DDGM Training with CelebA Dataset
===================================
Retrains the DDGM using CelebA (~200K images) instead of
the 5K assignment faces dataset. Uses the best hyperparameters
from the ablation study (LR=1e-4).

This addresses the data gap: the original dataset has only 5,000 images,
while papers typically use 50K-200K+ images for energy-based models.

Usage:
    python ddgm_celeba.py                         # full CelebA (~200K)
    python ddgm_celeba.py --max_samples 50000     # use 50K subset
    python ddgm_celeba.py --epochs 50             # override epochs
"""

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from celeba_dataset import CelebAFaceDataset
from dataset import GlassesDataset
from gan import Generator, weights_init
from ddgm import DeepEnergyModel, train_one_epoch, denorm
from evaluate import batch_fid, get_real_images

# ─────────────────────────────────────────────
#  Config — Best settings from ablation (Exp2)
# ─────────────────────────────────────────────

CONFIG = {
    'LATENT_DIM':       128,
    'BASE_FILTERS':     64,
    'NUM_EXPERTS':      512,
    'SIGMA':            1.0,
    'LR_DGM':           1e-4,      # Best from ablation
    'LR_DEM':           5e-5,      # TTUR: half of Generator LR for stability
    'D_STEPS':          1,         # 1:1 ratio — GP provides enough regularization
    'DROPOUT':          0.3,
    'G_ACTIVATION':     'relu',
    'GRADIENT_PENALTY': 10.0,
    'BATCH_SIZE':       64,        # Larger batch for more data
    'EPOCHS':           30,        # CelebA is bigger, fewer epochs needed
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DDGM training with CelebA')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit training set size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    if args.epochs:
        CONFIG['EPOCHS'] = args.epochs
    if args.batch_size:
        CONFIG['BATCH_SIZE'] = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: LR={CONFIG['LR_DGM']}, D_STEPS={CONFIG['D_STEPS']}, "
          f"EXPERTS={CONFIG['NUM_EXPERTS']}, EPOCHS={CONFIG['EPOCHS']}")

    # ── CelebA training dataset ────────────────────────────────
    print("\n--- Loading CelebA for Training ---")
    train_dataset = CelebAFaceDataset(
        root='.',
        split='train',
        target_size=(64, 64),
        augment=True,
        max_samples=args.max_samples,
        use_torchvision=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── CelebA evaluation dataset for FID ──────────────────────
    print("\n--- Loading CelebA for Evaluation ---")
    eval_dataset = CelebAFaceDataset(
        root='.',
        split='valid',
        target_size=(64, 64),
        augment=False,
        use_torchvision=False,
    )

    # ── Models ─────────────────────────────────────────────────
    generator = Generator(
        latent_dim=CONFIG['LATENT_DIM'],
        base_filters=CONFIG['BASE_FILTERS'],
        activation=CONFIG['G_ACTIVATION'],
        dropout=CONFIG['DROPOUT'],
    ).to(device)

    dem = DeepEnergyModel(
        base_filters=CONFIG['BASE_FILTERS'],
        num_experts=CONFIG['NUM_EXPERTS'],
        sigma=CONFIG['SIGMA'],
        dropout=CONFIG['DROPOUT'],
    ).to(device)

    generator.apply(weights_init)
    dem.apply(weights_init)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in dem.parameters())
    print(f"\nGenerator params: {g_params:,}  DEM params: {d_params:,}")

    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=CONFIG['LR_DGM'], betas=(0.5, 0.999)
    )
    optimizer_DEM = optim.Adam(
        dem.parameters(),
        lr=CONFIG['LR_DEM'], betas=(0.5, 0.999),
        weight_decay=1e-4
    )

    # ── Output directory ───────────────────────────────────────
    output_dir = 'ddgm_celeba_results'
    os.makedirs(output_dir, exist_ok=True)

    fixed_noise = torch.randn(8, CONFIG['LATENT_DIM'], device=device)

    # ── Training ───────────────────────────────────────────────
    dem_losses = []
    gen_losses = []
    best_gen_loss = float('inf')

    print(f"\n{'=' * 60}")
    print(f"  Training DDGM on CelebA ({len(train_dataset)} images)")
    print(f"  {len(train_loader)} batches/epoch x {CONFIG['EPOCHS']} epochs")
    print(f"{'=' * 60}\n")

    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        dem_loss, gen_loss, e_real, e_fake = train_one_epoch(
            generator, dem,
            optimizer_G, optimizer_DEM,
            train_loader, device,
            CONFIG['LATENT_DIM'], CONFIG['D_STEPS'],
            CONFIG['GRADIENT_PENALTY']
        )

        dem_losses.append(dem_loss)
        gen_losses.append(gen_loss)

        print(f"Epoch {epoch}/{CONFIG['EPOCHS']}  "
              f"DEM: {dem_loss:.4f}  Gen: {gen_loss:.4f}  "
              f"E_real: {e_real:.2f}  E_fake: {e_fake:.2f}")

        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            torch.save(generator.state_dict(),
                       os.path.join(output_dir, 'generator_best.pth'))


        # Save progress every 5 epochs
        if epoch % 5 == 0 or epoch == CONFIG['EPOCHS']:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            fig, axes = plt.subplots(1, 8, figsize=(20, 3))
            for i, ax in enumerate(axes):
                ax.imshow(denorm(samples[i]))
                ax.axis('off')
            plt.suptitle(f'DDGM-CelebA - Epoch {epoch}', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f'progress_epoch{epoch}.png'), dpi=150)
            plt.close()
            generator.train()

    # ── Save loss curves ───────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(dem_losses, label='DEM Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('DEM Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(gen_losses, label='Gen Loss', color='tab:orange')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Generator Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=150)
    plt.close()

    # ── Load best generator for evaluation ─────────────────────
    generator.load_state_dict(
        torch.load(os.path.join(output_dir, 'generator_best.pth'),
                   map_location=device, weights_only=True)
    )
    generator.eval()

    # ── Generate final samples ─────────────────────────────────
    with torch.no_grad():
        z = torch.randn(16, CONFIG['LATENT_DIM'], device=device)
        samples = generator(z)

    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(denorm(samples[i]))
        ax.axis('off')
    plt.suptitle('DDGM-CelebA - Final Generated Faces', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_final.png'), dpi=150)
    plt.close()

    # ── FID Evaluation (against CelebA dataset) ────────────
    print("\n--- FID Evaluation ---")
    num_eval = 1000
    fid_scores = {}
    batch_size_eval = 100

    import gc

    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        print(f"  Evaluating {label_name}...")

        # Generate images in batches, move to CPU immediately
        with torch.no_grad():
            generated_list = []
            for _ in range(num_eval // batch_size_eval):
                z = torch.randn(batch_size_eval, CONFIG['LATENT_DIM'], device=device)
                generated_list.append(generator(z).cpu())
            generated = torch.cat(generated_list, dim=0)
            del generated_list

        # Load real images — check label before loading image (fast path)
        real_images = []
        for i, lbl in enumerate(eval_dataset.labels):
            if lbl == label:
                img, _ = eval_dataset[i]
                real_images.append(img)
            if len(real_images) >= num_eval:
                break

        if len(real_images) == 0:
            print(f"  Warning: No real images found for {label_name}, skipping.")
            continue

        real = torch.stack(real_images)  # keep on CPU
        del real_images
        perm = torch.randperm(real.size(0))
        real = real[perm]

        fid_val = batch_fid(generated, real, device)
        fid_scores[label_name] = fid_val
        print(f"  FID ({label_name}): {fid_val:.4f}")

        # Cleanup between iterations
        del generated, real
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fid_scores['Average'] = np.mean(list(fid_scores.values()))
    print(f"  FID (Average): {fid_scores['Average']:.4f}")

    # ── Save results ───────────────────────────────────────────
    result = {
        'dataset': 'CelebA',
        'train_size': len(train_dataset),
        'config': CONFIG,
        'best_gen_loss': best_gen_loss,
        'fid_glasses': fid_scores['Glasses'],
        'fid_no_glasses': fid_scores['No Glasses'],
        'fid_average': fid_scores['Average'],
    }

    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DDGM-CelebA Training Complete")
    print(f"{'=' * 60}")
    print(f"  Dataset:       CelebA ({len(train_dataset)} images)")
    print(f"  Epochs:        {CONFIG['EPOCHS']}")
    print(f"  Best Gen Loss: {best_gen_loss:.4f}")
    print(f"  FID (Avg):     {fid_scores['Average']:.4f}")
    print(f"  Results:       {output_dir}/")
    print(f"{'=' * 60}")

    # Compare with original
    print(f"\n  Comparison with 5K dataset:")
    print(f"  Original DDGM (5K, best ablation): FID ~ 451.60")
    print(f"  CelebA DDGM ({len(train_dataset)} imgs):  FID = {fid_scores['Average']:.2f}")


if __name__ == '__main__':
    main()
