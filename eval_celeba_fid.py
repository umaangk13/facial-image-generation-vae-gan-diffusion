import os
import torch
import numpy as np
from celeba_dataset import CelebAFaceDataset
from gan import Generator
from evaluate import batch_fid, get_real_images

# Configuration to match CelebA run
CONFIG = {
    'LATENT_DIM': 128,
    'BASE_FILTERS': 64,
    'G_ACTIVATION': 'relu',
    'DROPOUT': 0.3,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = os.path.join('ddgm_celeba_results', 'generator_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find model at {checkpoint_path}")
        return

    # Load generator
    generator = Generator(
        latent_dim=CONFIG['LATENT_DIM'],
        base_filters=CONFIG['BASE_FILTERS'],
        activation=CONFIG['G_ACTIVATION'],
        dropout=CONFIG['DROPOUT'],
    ).to(device)

    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Load dataset
    print("\n--- Loading CelebA Evaluation Set ---")
    eval_dataset = CelebAFaceDataset(
        root='.',
        split='valid',
        target_size=(64, 64),
        augment=False,
        use_torchvision=False,
    )

    # FID Evaluation
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
            print(f"Warning: No real images found for label {label_name}!")
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

    avg_fid = np.mean(list(fid_scores.values()))
    print(f"\n  Final FID (Average): {avg_fid:.4f}")

if __name__ == '__main__':
    main()
