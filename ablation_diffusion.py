"""
Conditional DDPM Ablation Study — Automated
=============================================
Trains the Diffusion model with 5 different hyperparameter configs,
computes SSIM for each, saves generated images, and prints a summary table.

Usage:
    python ablation_diffusion.py

NOTE: Diffusion training is slower than VAE/GAN. Each experiment trains
for 50 epochs by default. You can reduce EPOCHS in the baseline config
below if you want faster results (e.g. 20 epochs for a quick run).
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset
from diffusion import (build_unet, GaussianDiffusion,
                       train_one_epoch, denorm)
from evaluate import compute_ssim, batch_ssim, get_real_images


# ─────────────────────────────────────────────
#  Ablation Configs
# ─────────────────────────────────────────────

BASELINE = {
    'name':             'Baseline',
    'changed_param':    'None',
    'changed_value':    '-',
    'TIMESTEPS':        1000,
    'BETA_START':       1e-4,
    'BETA_END':         0.02,
    'LR':               2e-4,
    'BATCH_SIZE':       32,
    'BASE_DIM':         64,
    'DIM_MULTS':        (1, 2, 4, 8),
    'NUM_RES_BLOCKS':   2,
    'EPOCHS':           50,
}

EXPERIMENTS = [
    {**BASELINE,
     'name':          'Exp1: Timesteps 200',
     'changed_param': 'TIMESTEPS',
     'changed_value': '200',
     'TIMESTEPS':     200},

    {**BASELINE,
     'name':          'Exp2: Beta End 0.05',
     'changed_param': 'BETA_END',
     'changed_value': '0.05',
     'BETA_END':      0.05},

    {**BASELINE,
     'name':          'Exp3: LR 5e-4',
     'changed_param': 'LR',
     'changed_value': '5e-4',
     'LR':            5e-4},

    {**BASELINE,
     'name':          'Exp4: Base Dim 32',
     'changed_param': 'BASE_DIM',
     'changed_value': '32',
     'BASE_DIM':      32},

    {**BASELINE,
     'name':          'Exp5: Res Blocks 1',
     'changed_param': 'NUM_RES_BLOCKS',
     'changed_value': '1',
     'NUM_RES_BLOCKS': 1},
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
                        shuffle=True, num_workers=2, drop_last=True)

    # Build model
    model = build_unet(
        base_dim=config['BASE_DIM'],
        dim_mults=config['DIM_MULTS'],
        num_res_blocks=config['NUM_RES_BLOCKS'],
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=config['TIMESTEPS'],
        beta_start=config['BETA_START'],
        beta_end=config['BETA_END'],
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=config['LR'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['EPOCHS']
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train
    train_losses = []
    best_loss = float('inf')

    for epoch in range(1, config['EPOCHS'] + 1):
        train_loss = train_one_epoch(
            model, diffusion, loader, optimizer, device
        )
        train_losses.append(train_loss)
        scheduler.step()

        if epoch % 10 == 0 or epoch == config['EPOCHS']:
            print(f"  Epoch {epoch}/{config['EPOCHS']}  "
                  f"Loss: {train_loss:.6f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, 'diffusion_best.pth'))

    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'DDPM Loss — {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss.png'), dpi=100)
    plt.close()

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, 'diffusion_best.pth'),
                    map_location=device, weights_only=True)
    )
    model.eval()

    # Generate sample images (3 glasses + 3 no-glasses)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, (label, label_name) in enumerate([(1, 'Glasses'), (0, 'No Glasses')]):
        imgs = diffusion.generate(model, 3, label, device)
        for col in range(3):
            axes[row, col].imshow(denorm(imgs[col]))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(label_name, fontsize=10)
    plt.suptitle(f'DDPM — {name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'generated.png'), dpi=150)
    plt.close()

    # Compute SSIM
    num_eval = 50
    ssim_scores = {}
    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        generated = diffusion.generate(model, num_eval, label, device)
        real = get_real_images(eval_dataset, label, num_eval, device)
        perm = torch.randperm(real.size(0))
        real = real[perm]
        ssim_val = batch_ssim(generated, real)
        ssim_scores[label_name] = ssim_val

    ssim_scores['Average'] = np.mean(list(ssim_scores.values()))

    result = {
        'name': name,
        'changed_param': config['changed_param'],
        'changed_value': config['changed_value'],
        'best_loss': best_loss,
        'final_loss': train_losses[-1],
        'params': param_count,
        'ssim_glasses': ssim_scores['Glasses'],
        'ssim_no_glasses': ssim_scores['No Glasses'],
        'ssim_average': ssim_scores['Average'],
    }

    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Best Loss: {best_loss:.6f}")
    print(f"  SSIM — Glasses: {ssim_scores['Glasses']:.4f}  "
          f"No Glasses: {ssim_scores['No Glasses']:.4f}  "
          f"Avg: {ssim_scores['Average']:.4f}")
    print(f"  Saved to {exp_dir}")

    return result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = GlassesDataset(
        csv_path='final_train.csv', images_dir='images',
        split='all', target_size=(64, 64), augment=True,
    )
    eval_dataset = GlassesDataset(
        csv_path='final_train.csv', images_dir='images',
        split='all', target_size=(64, 64), augment=False,
    )

    output_dir = 'ablation_diffusion'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    
    # Try to load existing results first
    results_file = os.path.join(output_dir, 'all_results.json')
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
                # Create a lookup dictionary by experiment name
                existing_dict = {r['name']: r for r in existing_results}
                print(f"Loaded {len(existing_dict)} existing results from {results_file}")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            existing_dict = {}
    else:
        existing_dict = {}

    for config in ALL_CONFIGS:
        name = config['name']
        exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
        exp_result_file = os.path.join(exp_dir, 'result.json')
        
        # If this experiment already finished successfully, use the saved result
        if os.path.exists(exp_result_file) and name in existing_dict:
            print(f"\n{'═' * 60}")
            print(f"  ⏭️ SKIPPING: {name} (Already completed)")
            print(f"{'═' * 60}")
            all_results.append(existing_dict[name])
        else:
            # Otherwise, run it
            result = run_experiment(
                config, train_dataset, eval_dataset, device, output_dir
            )
            all_results.append(result)
            
            # Save progress incrementally just in case it crashes again
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # ── Summary table ────────────────────────────────────────
    print(f"\n\n{'═' * 85}")
    print(f"  DDPM ABLATION TABLE")
    print(f"{'═' * 85}")
    header = (f"  {'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
              f"{'Loss':>8} {'SSIM(G)':>8} {'SSIM(NG)':>9} {'SSIM(Avg)':>9}")
    print(header)
    print(f"  {'─' * 80}")
    for r in all_results:
        row = (f"  {r['name']:<28} {r['changed_param']:<16} "
               f"{r['changed_value']:<10} "
               f"{r['best_loss']:>8.6f} "
               f"{r['ssim_glasses']:>8.4f} "
               f"{r['ssim_no_glasses']:>9.4f} "
               f"{r['ssim_average']:>9.4f}")
        print(row)
    print(f"{'═' * 85}")

    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("DDPM ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'Loss':>8} {'SSIM(G)':>8} {'SSIM(NG)':>9} {'SSIM(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r['best_loss']:>8.6f} "
                    f"{r['ssim_glasses']:>8.4f} "
                    f"{r['ssim_no_glasses']:>9.4f} "
                    f"{r['ssim_average']:>9.4f}\n")
        f.write("=" * 85 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")
    print(f"  - ablation_table.txt  (text table)")
    print(f"  - all_results.json    (machine-readable)")
    print(f"  - Each experiment folder has: loss.png, generated.png, result.json")
