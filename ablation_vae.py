"""
CVAE Ablation Study — Automated
================================
Trains the CVAE with 5 different hyperparameter configs (one change at a time),
computes SSIM for each, saves generated images, and prints a summary table.

Usage:
    python ablation_vae.py
"""

import os
import sys
import json
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset
from vae import CVAE, vae_loss, train_one_epoch
from evaluate import batch_fid, get_real_images


# ─────────────────────────────────────────────
#  Ablation Configs
# ─────────────────────────────────────────────

# Baseline config
BASELINE = {
    'name':              'Baseline',
    'changed_param':     'None',
    'changed_value':     '-',
    'LATENT_DIM':        128,
    'KL_WEIGHT':         0.5,
    'CONV_KERNEL':       3,
    'ACTIVATION':        'relu',
    'USE_INTERP':        False,
    'BASE_FILTERS':      32,
    'LR':                1e-3,
    'BATCH_SIZE':        32,
    'EPOCHS':            30,
}

# 5 ablation experiments — each changes ONE parameter from baseline
EXPERIMENTS = [
    {**BASELINE,
     'name':          'Exp1: Latent Dim 32',
     'changed_param': 'LATENT_DIM',
     'changed_value': '32',
     'LATENT_DIM':    32},

    {**BASELINE,
     'name':          'Exp2: KL Weight 0.05',
     'changed_param': 'KL_WEIGHT',
     'changed_value': '0.05',
     'KL_WEIGHT':     0.05},

    {**BASELINE,
     'name':          'Exp3: Kernel 5',
     'changed_param': 'CONV_KERNEL',
     'changed_value': '5',
     'CONV_KERNEL':   5},

    {**BASELINE,
     'name':          'Exp4: LeakyReLU',
     'changed_param': 'ACTIVATION',
     'changed_value': 'leakyrelu',
     'ACTIVATION':    'leakyrelu'},

    {**BASELINE,
     'name':          'Exp5: Bilinear Upsample',
     'changed_param': 'USE_INTERP',
     'changed_value': 'True',
     'USE_INTERP':    True},
]

ALL_CONFIGS = [BASELINE] + EXPERIMENTS


# ─────────────────────────────────────────────
#  Helper: denormalise for saving images
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


# ─────────────────────────────────────────────
#  Run a single experiment
# ─────────────────────────────────────────────

def run_experiment(config, dataset, eval_dataset, device, output_dir):
    """Train CVAE with given config, evaluate SSIM, save images."""

    name = config['name']
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"  Changed: {config['changed_param']} = {config['changed_value']}")
    print(f"{'═' * 60}")

    # Create output dir for this experiment
    exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
    os.makedirs(exp_dir, exist_ok=True)

    # DataLoader
    loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'],
                        shuffle=True, num_workers=2)

    # Build model
    model = CVAE(
        latent_dim=config['LATENT_DIM'],
        base_filters=config['BASE_FILTERS'],
        conv_kernel=config['CONV_KERNEL'],
        activation=config['ACTIVATION'],
        use_interpolation=config['USE_INTERP'],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['LR'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train
    train_losses = []
    best_loss = float('inf')

    for epoch in range(1, config['EPOCHS'] + 1):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, loader, optimizer, device, config['KL_WEIGHT']
        )
        train_losses.append(train_loss)
        scheduler.step(train_loss)

        if epoch % 10 == 0 or epoch == config['EPOCHS']:
            print(f"  Epoch {epoch}/{config['EPOCHS']}  "
                  f"Loss: {train_loss:.4f}  "
                  f"Recon: {train_recon:.4f}  KL: {train_kl:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, 'vae_best.pth'))

    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'CVAE Loss — {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss.png'), dpi=100)
    plt.close()

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, 'vae_best.pth'),
                    map_location=device, weights_only=True)
    )
    model.eval()

    # Generate sample images (3 glasses + 3 no-glasses)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, (label, label_name) in enumerate([(1, 'Glasses'), (0, 'No Glasses')]):
        imgs = model.generate(3, label, device)
        for col in range(3):
            axes[row, col].imshow(denorm(imgs[col]))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(label_name, fontsize=10)
    plt.suptitle(f'CVAE — {name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'generated.png'), dpi=150)
    plt.close()

    # Compute FID
    num_eval = 50
    fid_scores = {}
    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        generated = model.generate(num_eval, label, device)
        real = get_real_images(eval_dataset, label, num_eval, device)
        perm = torch.randperm(real.size(0))
        real = real[perm]
        fid_val = batch_fid(generated, real, device)
        fid_scores[label_name] = fid_val

    fid_scores['Average'] = np.mean(list(fid_scores.values()))

    # Save results
    result = {
        'name': name,
        'changed_param': config['changed_param'],
        'changed_value': config['changed_value'],
        'final_loss': train_losses[-1],
        'best_loss': best_loss,
        'params': param_count,
        'fid_glasses': fid_scores['Glasses'],
        'fid_no_glasses': fid_scores['No Glasses'],
        'fid_average': fid_scores['Average'],
    }

    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Final Loss: {train_losses[-1]:.4f}")
    print(f"  FID — Glasses: {fid_scores['Glasses']:.4f}  "
          f"No Glasses: {fid_scores['No Glasses']:.4f}  "
          f"Avg: {fid_scores['Average']:.4f}")
    print(f"  Saved to {exp_dir}")

    return result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    train_dataset = GlassesDataset(
        csv_path='final_train.csv', images_dir='images',
        split='all', target_size=(64, 64), augment=True,
    )
    eval_dataset = GlassesDataset(
        csv_path='final_train.csv', images_dir='images',
        split='all', target_size=(64, 64), augment=False,
    )

    output_dir = 'ablation_vae'
    os.makedirs(output_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    for config in ALL_CONFIGS:
        result = run_experiment(
            config, train_dataset, eval_dataset, device, output_dir
        )
        all_results.append(result)

    # ── Print final summary table ────────────────────────────
    print(f"\n\n{'═' * 85}")
    print(f"  CVAE ABLATION TABLE")
    print(f"{'═' * 85}")
    header = (f"  {'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
              f"{'Loss':>8} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}")
    print(header)
    print(f"  {'─' * 80}")
    for r in all_results:
        row = (f"  {r['name']:<28} {r['changed_param']:<16} "
               f"{r['changed_value']:<10} "
               f"{r['best_loss']:>8.4f} "
               f"{r['fid_glasses']:>8.4f} "
               f"{r['fid_no_glasses']:>9.4f} "
               f"{r['fid_average']:>9.4f}")
        print(row)
    print(f"{'═' * 85}")

    # Save table to file
    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("CVAE ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'Loss':>8} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r['best_loss']:>8.4f} "
                    f"{r['fid_glasses']:>8.4f} "
                    f"{r['fid_no_glasses']:>9.4f} "
                    f"{r['fid_average']:>9.4f}\n")
        f.write("=" * 85 + "\n")

    # Save as JSON too
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")
    print(f"  - ablation_table.txt  (text table)")
    print(f"  - all_results.json    (machine-readable)")
    print(f"  - Each experiment folder has: loss.png, generated.png, result.json")
