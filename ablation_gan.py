"""
Unconditional GAN Ablation Study — Automated
==============================================
Trains the GAN with 5 different hyperparameter configs (one change at a time),
computes SSIM for each, saves generated images, and prints a summary table.

Usage:
    python ablation_gan.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset
from gan import (Generator, Discriminator, weights_init,
                 train_one_epoch)
from evaluate import compute_ssim, batch_ssim, get_real_images


# ─────────────────────────────────────────────
#  Ablation Configs
# ─────────────────────────────────────────────

BASELINE = {
    'name':             'Baseline',
    'changed_param':    'None',
    'changed_value':    '-',
    'LATENT_DIM':       128,
    'BASE_FILTERS':     64,
    'LR_G':             2e-4,
    'LR_D':             2e-4,
    'D_STEPS':          1,
    'LABEL_SMOOTHING':  0.1,
    'DROPOUT':          0.3,
    'G_ACTIVATION':     'relu',
    'BATCH_SIZE':       32,
    'EPOCHS':           50,
}

EXPERIMENTS = [
    {**BASELINE,
     'name':          'Exp1: D_Steps 3',
     'changed_param': 'D_STEPS',
     'changed_value': '3',
     'D_STEPS':       3},

    {**BASELINE,
     'name':          'Exp2: No Label Smooth',
     'changed_param': 'LABEL_SMOOTHING',
     'changed_value': '0.0',
     'LABEL_SMOOTHING': 0.0},

    {**BASELINE,
     'name':          'Exp3: Dropout 0.5',
     'changed_param': 'DROPOUT',
     'changed_value': '0.5',
     'DROPOUT':       0.5},

    {**BASELINE,
     'name':          'Exp4: LR 1e-4',
     'changed_param': 'LR_G / LR_D',
     'changed_value': '1e-4',
     'LR_G':          1e-4,
     'LR_D':          1e-4},

    {**BASELINE,
     'name':          'Exp5: LeakyReLU Gen',
     'changed_param': 'G_ACTIVATION',
     'changed_value': 'leakyrelu',
     'G_ACTIVATION':  'leakyrelu'},
]

ALL_CONFIGS = [BASELINE] + EXPERIMENTS


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


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

    discriminator = Discriminator(
        base_filters=config['BASE_FILTERS'],
        dropout=config['DROPOUT'],
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator params: {g_params:,}  Discriminator params: {d_params:,}")

    optimizer_G = optim.Adam(generator.parameters(),
                              lr=config['LR_G'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                              lr=config['LR_D'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Train
    d_losses = []
    g_losses = []
    best_g_loss = float('inf')

    for epoch in range(1, config['EPOCHS'] + 1):
        d_loss, g_loss, d_acc = train_one_epoch(
            generator, discriminator,
            optimizer_G, optimizer_D,
            loader, criterion, device,
            config['LATENT_DIM'], config['D_STEPS'],
            config['LABEL_SMOOTHING']
        )
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % 10 == 0 or epoch == config['EPOCHS']:
            print(f"  Epoch {epoch}/{config['EPOCHS']}  "
                  f"D_loss: {d_loss:.4f}  G_loss: {g_loss:.4f}  "
                  f"D_acc: {d_acc:.2f}")

        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save(generator.state_dict(),
                       os.path.join(exp_dir, 'generator_best.pth'))

    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='D Loss')
    plt.plot(g_losses, label='G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'GAN Loss — {name}')
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

    # Generate sample images (8 random faces — unconditional)
    with torch.no_grad():
        z = torch.randn(8, config['LATENT_DIM'], device=device)
        samples = generator(z)

    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for i, ax in enumerate(axes):
        ax.imshow(denorm(samples[i]))
        ax.axis('off')
    plt.suptitle(f'GAN — {name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'generated.png'), dpi=150)
    plt.close()

    # Compute SSIM (unconditional — label is ignored by GAN)
    num_eval = 50
    ssim_scores = {}
    for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
        with torch.no_grad():
            z = torch.randn(num_eval, config['LATENT_DIM'], device=device)
            generated = generator(z)
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
        'best_g_loss': best_g_loss,
        'final_d_loss': d_losses[-1],
        'final_g_loss': g_losses[-1],
        'g_params': g_params,
        'd_params': d_params,
        'ssim_glasses': ssim_scores['Glasses'],
        'ssim_no_glasses': ssim_scores['No Glasses'],
        'ssim_average': ssim_scores['Average'],
    }

    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Best G Loss: {best_g_loss:.4f}")
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

    output_dir = 'ablation_gan'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for config in ALL_CONFIGS:
        result = run_experiment(
            config, train_dataset, eval_dataset, device, output_dir
        )
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────
    print(f"\n\n{'═' * 85}")
    print(f"  GAN ABLATION TABLE")
    print(f"{'═' * 85}")
    header = (f"  {'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
              f"{'G Loss':>8} {'SSIM(G)':>8} {'SSIM(NG)':>9} {'SSIM(Avg)':>9}")
    print(header)
    print(f"  {'─' * 80}")
    for r in all_results:
        row = (f"  {r['name']:<28} {r['changed_param']:<16} "
               f"{r['changed_value']:<10} "
               f"{r['best_g_loss']:>8.4f} "
               f"{r['ssim_glasses']:>8.4f} "
               f"{r['ssim_no_glasses']:>9.4f} "
               f"{r['ssim_average']:>9.4f}")
        print(row)
    print(f"{'═' * 85}")

    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("GAN ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'G Loss':>8} {'SSIM(G)':>8} {'SSIM(NG)':>9} {'SSIM(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r['best_g_loss']:>8.4f} "
                    f"{r['ssim_glasses']:>8.4f} "
                    f"{r['ssim_no_glasses']:>9.4f} "
                    f"{r['ssim_average']:>9.4f}\n")
        f.write("=" * 85 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")
