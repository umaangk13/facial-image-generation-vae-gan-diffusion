"""
Compute FID scores from saved model checkpoints for all three ablation studies
(Diffusion, VAE, GAN) and update ablation_table.txt + all_results.json.

Usage:
    python eval_fid_from_checkpoints.py
"""

import os
import json
import torch
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset
from evaluate import batch_fid, get_real_images

NUM_EVAL = 50


# ─────────────────────────────────────────────
#  Diffusion
# ─────────────────────────────────────────────

def eval_diffusion(eval_dataset, device):
    from diffusion import build_unet, GaussianDiffusion
    from ablation_diffusion import ALL_CONFIGS

    output_dir = 'ablation_diffusion'
    results_path = os.path.join(output_dir, 'all_results.json')
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing = {r['name']: r for r in json.load(f)}

    all_results = []
    for config in ALL_CONFIGS:
        name = config['name']
        exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
        model_path = os.path.join(exp_dir, 'diffusion_best.pth')

        if not os.path.exists(model_path):
            print(f"  [SKIP] {name} — no checkpoint")
            if name in existing:
                all_results.append(existing[name])
            continue

        print(f"  Evaluating {name}...")
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
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        fid_scores = {}
        for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
            generated = diffusion.generate(model, NUM_EVAL, label, device)
            real = get_real_images(eval_dataset, label, NUM_EVAL, device)
            real = real[torch.randperm(real.size(0))]
            fid_scores[label_name] = batch_fid(generated, real, device)
            print(f"    FID ({label_name}): {fid_scores[label_name]:.4f}")
        fid_scores['Average'] = float(np.mean(list(fid_scores.values())))

        result = dict(existing.get(name, {
            'name': name,
            'changed_param': config['changed_param'],
            'changed_value': config['changed_value'],
            'best_loss': 0.0, 'final_loss': 0.0,
            'params': sum(p.numel() for p in model.parameters()),
        }))
        for k in ['ssim_glasses', 'ssim_no_glasses', 'ssim_average']:
            result.pop(k, None)
        result['fid_glasses'] = float(fid_scores['Glasses'])
        result['fid_no_glasses'] = float(fid_scores['No Glasses'])
        result['fid_average'] = float(fid_scores['Average'])

        with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    # Write table
    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("DDPM ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'Loss':>8} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r.get('best_loss', 0.0):>8.6f} "
                    f"{r.get('fid_glasses', 0.0):>8.4f} "
                    f"{r.get('fid_no_glasses', 0.0):>9.4f} "
                    f"{r.get('fid_average', 0.0):>9.4f}\n")
        f.write("=" * 85 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"  ✅ Saved to {output_dir}/\n")


# ─────────────────────────────────────────────
#  VAE
# ─────────────────────────────────────────────

def eval_vae(eval_dataset, device):
    from vae import CVAE
    from ablation_vae import ALL_CONFIGS

    output_dir = 'ablation_vae'
    results_path = os.path.join(output_dir, 'all_results.json')
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing = {r['name']: r for r in json.load(f)}

    all_results = []
    for config in ALL_CONFIGS:
        name = config['name']
        exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
        model_path = os.path.join(exp_dir, 'vae_best.pth')

        if not os.path.exists(model_path):
            print(f"  [SKIP] {name} — no checkpoint")
            if name in existing:
                all_results.append(existing[name])
            continue

        print(f"  Evaluating {name}...")
        model = CVAE(
            latent_dim=config['LATENT_DIM'],
            base_filters=config['BASE_FILTERS'],
            conv_kernel=config['CONV_KERNEL'],
            activation=config['ACTIVATION'],
            use_interpolation=config['USE_INTERP'],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        fid_scores = {}
        for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
            generated = model.generate(NUM_EVAL, label, device)
            real = get_real_images(eval_dataset, label, NUM_EVAL, device)
            real = real[torch.randperm(real.size(0))]
            fid_scores[label_name] = batch_fid(generated, real, device)
            print(f"    FID ({label_name}): {fid_scores[label_name]:.4f}")
        fid_scores['Average'] = float(np.mean(list(fid_scores.values())))

        result = dict(existing.get(name, {
            'name': name,
            'changed_param': config['changed_param'],
            'changed_value': config['changed_value'],
            'best_loss': 0.0, 'final_loss': 0.0,
            'params': sum(p.numel() for p in model.parameters()),
        }))
        for k in ['ssim_glasses', 'ssim_no_glasses', 'ssim_average']:
            result.pop(k, None)
        result['fid_glasses'] = float(fid_scores['Glasses'])
        result['fid_no_glasses'] = float(fid_scores['No Glasses'])
        result['fid_average'] = float(fid_scores['Average'])

        with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    # Write table
    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("CVAE ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'Loss':>8} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r.get('best_loss', 0.0):>8.4f} "
                    f"{r.get('fid_glasses', 0.0):>8.4f} "
                    f"{r.get('fid_no_glasses', 0.0):>9.4f} "
                    f"{r.get('fid_average', 0.0):>9.4f}\n")
        f.write("=" * 85 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"  ✅ Saved to {output_dir}/\n")


# ─────────────────────────────────────────────
#  GAN
# ─────────────────────────────────────────────

def eval_gan(eval_dataset, device):
    from gan import Generator
    from ablation_gan import ALL_CONFIGS

    output_dir = 'ablation_gan'
    results_path = os.path.join(output_dir, 'all_results.json')
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing = {r['name']: r for r in json.load(f)}

    all_results = []
    for config in ALL_CONFIGS:
        name = config['name']
        exp_dir = os.path.join(output_dir, name.replace(': ', '_').replace(' ', '_'))
        model_path = os.path.join(exp_dir, 'generator_best.pth')

        if not os.path.exists(model_path):
            print(f"  [SKIP] {name} — no checkpoint")
            if name in existing:
                all_results.append(existing[name])
            continue

        print(f"  Evaluating {name}...")
        generator = Generator(
            latent_dim=config['LATENT_DIM'],
            base_filters=config['BASE_FILTERS'],
            activation=config['G_ACTIVATION'],
            dropout=config['DROPOUT'],
        ).to(device)
        generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        generator.eval()

        fid_scores = {}
        for label, label_name in [(1, 'Glasses'), (0, 'No Glasses')]:
            with torch.no_grad():
                z = torch.randn(NUM_EVAL, config['LATENT_DIM'], device=device)
                generated = generator(z)
            real = get_real_images(eval_dataset, label, NUM_EVAL, device)
            real = real[torch.randperm(real.size(0))]
            fid_scores[label_name] = batch_fid(generated, real, device)
            print(f"    FID ({label_name}): {fid_scores[label_name]:.4f}")
        fid_scores['Average'] = float(np.mean(list(fid_scores.values())))

        result = dict(existing.get(name, {
            'name': name,
            'changed_param': config['changed_param'],
            'changed_value': config['changed_value'],
            'best_g_loss': 0.0, 'final_d_loss': 0.0, 'final_g_loss': 0.0,
            'g_params': sum(p.numel() for p in generator.parameters()),
            'd_params': 0,
        }))
        for k in ['ssim_glasses', 'ssim_no_glasses', 'ssim_average']:
            result.pop(k, None)
        result['fid_glasses'] = float(fid_scores['Glasses'])
        result['fid_no_glasses'] = float(fid_scores['No Glasses'])
        result['fid_average'] = float(fid_scores['Average'])

        with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    # Write table
    loss_key = 'best_g_loss'
    with open(os.path.join(output_dir, 'ablation_table.txt'), 'w') as f:
        f.write("GAN ABLATION TABLE\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Experiment':<28} {'Changed Param':<16} {'Value':<10} "
                f"{'G Loss':>8} {'FID(G)':>8} {'FID(NG)':>9} {'FID(Avg)':>9}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['name']:<28} {r['changed_param']:<16} "
                    f"{r['changed_value']:<10} "
                    f"{r.get(loss_key, 0.0):>8.4f} "
                    f"{r.get('fid_glasses', 0.0):>8.4f} "
                    f"{r.get('fid_no_glasses', 0.0):>9.4f} "
                    f"{r.get('fid_average', 0.0):>9.4f}\n")
        f.write("=" * 85 + "\n")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"  ✅ Saved to {output_dir}/\n")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    eval_dataset = GlassesDataset(
        csv_path='final_train.csv', images_dir='images',
        split='all', target_size=(64, 64), augment=False,
    )

    # # ── Diffusion (already done) ──
    # print("\n══════════════════════════════════════")
    # print("  DIFFUSION — Computing FID")
    # print("══════════════════════════════════════")
    # eval_diffusion(eval_dataset, device)

    # # ── VAE (run ablation_vae.py directly instead) ──
    # print("══════════════════════════════════════")
    # print("  VAE — Computing FID")
    # print("══════════════════════════════════════")
    # eval_vae(eval_dataset, device)

    print("══════════════════════════════════════")
    print("  GAN — Computing FID")
    print("══════════════════════════════════════")
    eval_gan(eval_dataset, device)

    print("\n🎉 All ablation tables updated with FID scores!")
