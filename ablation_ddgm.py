"""
DDGM Ablation Study — Automated
=================================
Trains the DDGM with a baseline + 3 hyperparameter variants (one change at a time),
computes FID for each, saves generated images, and prints a summary table.

Variants:
  Baseline:  NUM_EXPERTS=512, LR=2e-4, LATENT_DIM=128
  Exp1:      NUM_EXPERTS=256  (fewer experts)
  Exp2:      LR=1e-4          (lower learning rate)
  Exp3:      LATENT_DIM=256   (larger latent space)

Usage:
    python ablation_ddgm.py
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
from gan import Generator, weights_init
from ddgm import DeepEnergyModel, train_one_epoch, denorm
from evaluate import batch_fid, get_real_images


# ─────────────────────────────────────────────
#  Ablation Configs
# ─────────────────────────────────────────────

BASELINE = {
    'name':               'Baseline',
    'changed_param':      'None',
    'changed_value':      '-',
    'LATENT_DIM':         128,
    'BASE_FILTERS':       64,
    'NUM_EXPERTS':        512,
    'SIGMA':              1.0,
    'LR_DGM':             2e-4,
    'LR_DEM':             2e-4,
    'D_STEPS':            1,
    'DROPOUT':            0.3,
    'G_ACTIVATION':       'relu',
    'GRADIENT_PENALTY':   10.0,
    'BATCH_SIZE':         32,
    'EPOCHS':             50,
}

EXPERIMENTS = [
    {**BASELINE,
     'name':          'Exp1: 256 Experts',
     'changed_param': 'NUM_EXPERTS',
     'changed_value': '256',
     'NUM_EXPERTS':   256},

    {**BASELINE,
     'name':          'Exp2: LR 1e-4',
     'changed_param': 'LR_DGM / LR_DEM',
     'changed_value': '1e-4',
     'LR_DGM':        1e-4,
     'LR_DEM':        1e-4},

    {**BASELINE,
     'name':          'Exp3: Latent 256',
     'changed_param': 'LATENT_DIM',
     'changed_value': '256',
     'LATENT_DIM':    256},
]

ALL_CONFIGS = [BASELINE] + EXPERIMENTS


# ─────────────────────────────────────────────
#  Run a single experiment
# ─────────────────────────────────────────────

def run_experiment(config, dataset, eval_dataset, device, output_dir):
    name = config['name']
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"  Changed: {config['changed_param']} = {config['changed_value']}")
    print(f"{'═' * 60}")

    exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
    os.makedirs(exp_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'],
                        shuffle=True, num_workers=2)

    # Build models
    generator = Generator(
        latent_dim=config['LATENT_DIM'],
        base_filters=config['BASE_FILTERS'],
        activation=config['G_ACTIVATION'],
        dropout=config['DROPOUT'],
    ).to(device)

    dem = DeepEnergyModel(
        base_filters=config['BASE_FILTERS'],
        num_experts=config['NUM_EXPERTS'],
        sigma=config['SIGMA'],
        dropout=config['DROPOUT'],
    ).to(device)

    generator.apply(weights_init)
    dem.apply(weights_init)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in dem.parameters())
    print(f"  Generator params: {g_params:,}  DEM params: {d_params:,}")

    optimizer_G = optim.Adam(generator.parameters(),
                              lr=config['LR_DGM'], betas=(0.5, 0.999))
    optimizer_DEM = optim.Adam(dem.parameters(),
                                lr=config['LR_DEM'], betas=(0.5, 0.999))

    # Train
    dem_losses = []
    gen_losses = []
    best_gen_loss = float('inf')

    for epoch in range(1, config['EPOCHS'] + 1):
        dem_loss, gen_loss, e_real, e_fake = train_one_epoch(
            generator, dem,
            optimizer_G, optimizer_DEM,
            loader, device,
            config['LATENT_DIM'], config['D_STEPS'],
            config['GRADIENT_PENALTY']
        )
        dem_losses.append(dem_loss)
        gen_losses.append(gen_loss)

        if epoch % 10 == 0 or epoch == config['EPOCHS']:
            print(f"  Epoch {epoch}/{config['EPOCHS']}  "
                  f"DEM_loss: {dem_loss:.4f}  Gen_loss: {gen_loss:.4f}  "
                  f"E_real: {e_real:.2f}  E_fake: {e_fake:.2f}")

        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            torch.save(generator.state_dict(),
                       os.path.join(exp_dir, 'generator_best.pth'))

    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(dem_losses, label='DEM Loss')
    plt.plot(gen_losses, label='Gen Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'DDGM Loss — {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss.png'), dpi=100)
    plt.close()

    # Load best generator
    generator.load_state_dict(
        torch.load(os.path.join(exp_dir, 'generator_best.pth'),
                    map_location=device, weights_only=True)
    )
    generator.eval()

    # Generate sample images
    with torch.no_grad():
        z = torch.randn(8, config['LATENT_DIM'], device=device)
        samples = generator(z)

    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for i, ax in enumerate(axes):
        ax.imshow(denorm(samples[i]))
        ax.axis('off')
    plt.suptitle(f'DDGM — {name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'generated.png'), dpi=150)
    plt.close()

    # Compute FID
    num_eval = 1000
    fid_scores = {}
    batch_size_eval = 100
    
    import gc

    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        print(f"  Evaluating {label_name}...")
        with torch.no_grad():
            generated_list = []
            for _ in range(num_eval // batch_size_eval):
                z = torch.randn(batch_size_eval, config['LATENT_DIM'], device=device)
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

    result = {
        'name': name,
        'changed_param': config['changed_param'],
        'changed_value': config['changed_value'],
        'best_gen_loss': best_gen_loss,
        'final_dem_loss': dem_losses[-1],
        'final_gen_loss': gen_losses[-1],
        'g_params': g_params,
        'd_params': d_params,
        'fid_glasses': fid_scores['Glasses'],
        'fid_no_glasses': fid_scores['No Glasses'],
        'fid_average': fid_scores['Average'],
    }

    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Best Gen Loss: {best_gen_loss:.4f}")
    print(f"  FID — Glasses: {fid_scores['Glasses']:.4f}  "
          f"No Glasses: {fid_scores['No Glasses']:.4f}  "
          f"Avg: {fid_scores['Average']:.4f}")
    print(f"  Saved to {exp_dir}")

    return result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DDGM Ablation study with CelebA')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit training set size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs for all experiments')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for all experiments')
    args = parser.parse_args()

    if args.epochs is not None:
        for config in ALL_CONFIGS:
            config['EPOCHS'] = args.epochs
    if args.batch_size is not None:
        for config in ALL_CONFIGS:
            config['BATCH_SIZE'] = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n--- Loading CelebA for Training ---")
    train_dataset = CelebAFaceDataset(
        root='.',
        split='train',
        target_size=(64, 64),
        augment=True,
        max_samples=args.max_samples,
        use_torchvision=False,
    )

    print("\n--- Loading CelebA for Evaluation ---")
    eval_dataset = CelebAFaceDataset(
        root='.',
        split='valid',
        target_size=(64, 64),
        augment=False,
        use_torchvision=False,
    )

    output_dir = 'ablation_ddgm'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for config in ALL_CONFIGS:
        result = run_experiment(
            config, train_dataset, eval_dataset, device, output_dir
        )
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────
    print(f"\n\n{'═' * 90}")
    print(f"  DDGM ABLATION TABLE")
    print(f"{'═' * 90}")
    header = (f"  {'Experiment':<28} {'Changed Param':<18} {'Value':<10} "
              f"{'Gen Loss':>9} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}")
    print(header)
    print(f"  {'─' * 85}")
    for r in all_results:
        row = (f"  {r['name']:<28} {r['changed_param']:<18} "
               f"{r['changed_value']:<10} "
               f"{r['best_gen_loss']:>9.4f} "
               f"{r['fid_glasses']:>8.4f} "
               f"{r['fid_no_glasses']:>9.4f} "
               f"{r['fid_average']:>9.4f}")
        print(row)
    print(f"{'═' * 90}")

    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("DDGM ABLATION TABLE\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<18} {'Value':<10} "
                f"{'Gen Loss':>9} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<18} "
                    f"{r['changed_value']:<10} "
                    f"{r['best_gen_loss']:>9.4f} "
                    f"{r['fid_glasses']:>8.4f} "
                    f"{r['fid_no_glasses']:>9.4f} "
                    f"{r['fid_average']:>9.4f}\n")
        f.write("=" * 90 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")
