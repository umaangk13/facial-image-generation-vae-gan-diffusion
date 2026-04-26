"""
Deep Directed Generative Model (DDGM) — Assignment 2
=====================================================
Implements the framework from:
  "Deep Directed Generative Model with Energy-Based Probability Estimation"
  (Kim & Bengio, 2016) — https://arxiv.org/abs/1606.03439

The standard GAN discriminator is replaced with a Deep Energy Model (DEM)
that assigns scalar energy values to inputs. The generator (Deep Generative
Model / DGM) is trained to produce samples that the energy model assigns
low energy to.

Energy function (Eq. 11 from paper):
  E(x) = (1/σ²) xᵀx − bᵀx − Σᵢ log(1 + exp(Wᵢᵀ f(x) + bᵢ))

Training objectives:
  DEM: minimize E[E(x_real)] − E[E(x_fake)]   (Eq. 9)
  DGM: minimize E[E(G(z))]                     (Eq. 10)

Usage:
    python ddgm.py
"""

import os
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
from gan import Generator, ResBlock, weights_init


# ─────────────────────────────────────────────
#  Deep Energy Model (DEM)
#  Replaces the standard discriminator with an
#  energy function based on product of experts.
# ─────────────────────────────────────────────

class DeepEnergyModel(nn.Module):
    """
    Deep Energy Model from Kim & Bengio (2016).

    Architecture:
        1. Feature extractor f_φ(x): Conv layers extracting high-level features
        2. Energy function E(x) defined as product of experts (Eq. 11):
           E(x) = (1/σ²) xᵀx − bᵀx − Σᵢ log(1 + exp(Wᵢᵀ f(x) + bᵢ))

    Hyperparameters (for ablation):
        base_filters : base number of conv filters (try 64, 128)
        num_experts  : number of expert units (try 256, 512, 1024)
        sigma        : global variance parameter (try 0.5, 1.0, 2.0)
        dropout      : dropout rate in feature extractor
        leaky_slope  : LeakyReLU negative slope
    """

    def __init__(
        self,
        base_filters=64,
        num_experts=512,
        sigma=1.0,
        dropout=0.3,
        leaky_slope=0.2,
        energy_clamp=1000.0,
    ):
        super().__init__()

        self.sigma = sigma
        self.num_experts = num_experts
        self.energy_clamp = energy_clamp  # clamp energy magnitude for stability

        # ── Feature extractor f_φ(x) ─────────────────────────
        # Same conv backbone as GAN discriminator:
        # (3,64,64) → (f,32,32) → (2f,16,16) → (4f,8,8) → (8f,4,4)
        self.feature_extractor = nn.Sequential(
            ResBlock(3, base_filters, 4, 2, 1, leaky_slope, dropout, use_bn=False),
            ResBlock(base_filters, base_filters * 2, 4, 2, 1, leaky_slope, dropout),
            ResBlock(base_filters * 2, base_filters * 4, 4, 2, 1, leaky_slope, dropout),
            ResBlock(base_filters * 4, base_filters * 8, 4, 2, 1, leaky_slope, dropout),
        )

        feature_dim = base_filters * 8 * 4 * 4  # flattened feature size
        self.input_dim = 3 * 64 * 64             # flattened image size

        # ── Energy function parameters ────────────────────────
        # b vector for the bᵀx term (mean bias over input data)
        self.b_input = nn.Parameter(torch.zeros(self.input_dim))

        # Expert weights W and biases b for product of experts
        # Each expert: −log(1 + exp(Wᵢᵀ f(x) + bᵢ))
        self.expert_W = nn.utils.spectral_norm(
            nn.Linear(feature_dim, num_experts, bias=True)
        )

    def forward(self, x):
        """
        Compute energy E(x) for input images.

        Args:
            x: (B, 3, 64, 64) image tensor

        Returns:
            energy: (B,) scalar energy for each image
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (B, 3*64*64)

        # Term 1: (1/σ²) xᵀx — captures global variance
        # STABILITY FIX: normalize by input_dim to keep magnitude
        # comparable to other terms (prevents ~3000 bias for 64x64)
        term_xTx = (1.0 / (self.sigma ** 2)) * \
            (x_flat * x_flat).sum(dim=1) / self.input_dim

        # Term 2: −bᵀx — captures mean (also normalized)
        term_bx = -(self.b_input * x_flat).sum(dim=1) / self.input_dim

        # Term 3: −Σᵢ log(1 + exp(Wᵢᵀ f(x) + bᵢ)) — product of experts
        # (normalized by num_experts to keep scale consistent)
        features = self.feature_extractor(x)           # (B, 8f, 4, 4)
        features_flat = features.view(batch_size, -1)  # (B, feature_dim)
        expert_activations = self.expert_W(features_flat)  # (B, num_experts)
        term_experts = -nn.functional.softplus(expert_activations).sum(dim=1) \
            / self.num_experts

        energy = term_xTx + term_bx + term_experts  # (B,)

        # STABILITY FIX: soft clamp via tanh (gradient never fully vanishes)
        energy = self.energy_clamp * torch.tanh(energy / self.energy_clamp)

        return energy


# ─────────────────────────────────────────────
#  Training Steps
# ─────────────────────────────────────────────

def train_energy_model(dem, generator, real_images, optimizer_DEM,
                       device, latent_dim, gradient_penalty_weight=0.0):
    """
    Train the Deep Energy Model (Eq. 9):
    Push energy DOWN on real samples, push energy UP on generated samples.

    L_DEM = E[E(x_real)] − E[E(x_fake)]
    """
    dem.train()
    batch_size = real_images.size(0)

    optimizer_DEM.zero_grad()

    # Energy on real data — push DOWN
    energy_real = dem(real_images)

    # Energy on generated data — push UP
    z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        fake_images = generator(z)
    energy_fake = dem(fake_images)

    # Loss: minimize E[E(real)] - E[E(fake)]
    loss_dem = energy_real.mean() - energy_fake.mean()

    # Optional: gradient penalty for stability
    if gradient_penalty_weight > 0:
        gp = _gradient_penalty(dem, real_images, fake_images, device)
        loss_dem = loss_dem + gradient_penalty_weight * gp

    loss_dem.backward()
    # STABILITY FIX: clip gradient norms to prevent explosive updates
    torch.nn.utils.clip_grad_norm_(dem.parameters(), max_norm=1.0)
    optimizer_DEM.step()

    return loss_dem.item(), energy_real.mean().item(), energy_fake.mean().item()


def train_generator_ddgm(generator, dem, optimizer_G,
                          batch_size, latent_dim, device):
    """
    Train the Deep Generative Model (Eq. 10):
    Generate samples and minimize the energy assigned by DEM.

    L_DGM = E[E(G(z))]
    """
    generator.train()

    optimizer_G.zero_grad()

    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    energy_fake = dem(fake_images)

    # Loss: minimize energy of generated samples
    loss_gen = energy_fake.mean()

    loss_gen.backward()
    # STABILITY FIX: clip gradient norms for generator too
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer_G.step()

    return loss_gen.item()


def _gradient_penalty(dem, real_images, fake_images, device):
    """Gradient penalty for energy model stability."""
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    energy_interp = dem(interpolated)
    gradients = torch.autograd.grad(
        outputs=energy_interp.sum(),
        inputs=interpolated,
        create_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_one_epoch(generator, dem, optimizer_G, optimizer_DEM,
                    dataloader, device, latent_dim, d_steps=1,
                    gradient_penalty_weight=0.0):
    """Train one epoch of DDGM."""
    total_dem_loss = 0.0
    total_gen_loss = 0.0
    total_e_real = 0.0
    total_e_fake = 0.0
    n_batches = 0

    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # ── Train DEM (energy model) ──────────────────────────
        for _ in range(d_steps):
            dem_loss, e_real, e_fake = train_energy_model(
                dem, generator, real_images, optimizer_DEM,
                device, latent_dim, gradient_penalty_weight
            )

        # ── Train DGM (generator) ─────────────────────────────
        gen_loss = train_generator_ddgm(
            generator, dem, optimizer_G,
            batch_size, latent_dim, device
        )

        total_dem_loss += dem_loss
        total_gen_loss += gen_loss
        total_e_real += e_real
        total_e_fake += e_fake
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"DEM_loss: {dem_loss:.4f}  Gen_loss: {gen_loss:.4f}  "
                  f"E_real: {e_real:.2f}  E_fake: {e_fake:.2f}")

    return (total_dem_loss / n_batches,
            total_gen_loss / n_batches,
            total_e_real / n_batches,
            total_e_fake / n_batches)


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


def visualise_random(generator, device, latent_dim, n=8, save_path=None):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        imgs = generator(z)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, ax in enumerate(axes):
        ax.imshow(denorm(imgs[i]))
        ax.axis('off')

    plt.suptitle("DDGM — Random Generation", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def plot_losses(dem_losses, gen_losses, save_path=None):
    epochs = range(1, len(dem_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, dem_losses, label='DEM Loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Deep Energy Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, gen_losses, label='Generator Loss', color='tab:orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Generator Loss (Energy of Fakes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def plot_energies(e_reals, e_fakes, save_path=None):
    """Plot mean energy of real vs fake samples over training."""
    epochs = range(1, len(e_reals) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, e_reals, label='E(real)', color='tab:green')
    plt.plot(epochs, e_fakes, label='E(fake)', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Energy')
    plt.title('DDGM — Energy of Real vs Fake Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── Config ───────────────────────────────────────────────────
    CSV_PATH              = 'final_train.csv'
    IMAGES_DIR            = 'images'
    EPOCHS                = 50
    BATCH_SIZE            = 32
    LATENT_DIM            = 128
    BASE_FILTERS          = 64
    NUM_EXPERTS           = 512
    SIGMA                 = 1.0
    LR_DGM                = 2e-4
    LR_DEM                = 2e-4
    D_STEPS               = 1
    DROPOUT               = 0.3
    G_ACTIVATION          = 'relu'
    GRADIENT_PENALTY      = 10.0   # GP weight for DEM stability

    RESULTS_DIR = 'ddgm_results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset ──────────────────────────────────────────────────
    full_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='all', target_size=(64, 64), augment=True,
    )
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)

    print(f"Training on {len(full_dataset)} images")

    # ── Models ───────────────────────────────────────────────────
    # Reuse GAN generator (DGM)
    generator = Generator(
        latent_dim=LATENT_DIM,
        base_filters=BASE_FILTERS,
        activation=G_ACTIVATION,
        dropout=DROPOUT,
    ).to(device)

    # New energy-based discriminator (DEM)
    dem = DeepEnergyModel(
        base_filters=BASE_FILTERS,
        num_experts=NUM_EXPERTS,
        sigma=SIGMA,
        dropout=DROPOUT,
    ).to(device)

    generator.apply(weights_init)
    dem.apply(weights_init)

    print(f"Generator (DGM) params : "
          f"{sum(p.numel() for p in generator.parameters()):,}")
    print(f"Energy Model (DEM) params : "
          f"{sum(p.numel() for p in dem.parameters()):,}")

    # ── Optimisers ───────────────────────────────────────────────
    optimizer_G = optim.Adam(generator.parameters(),
                              lr=LR_DGM, betas=(0.5, 0.999))
    optimizer_DEM = optim.Adam(dem.parameters(),
                                lr=LR_DEM, betas=(0.5, 0.999))

    # Fixed noise to track progress across epochs
    fixed_noise = torch.randn(8, LATENT_DIM, device=device)

    # ── Forward pass check ───────────────────────────────────────
    sample_imgs, _ = next(iter(train_loader))
    sample_imgs = sample_imgs.to(device)
    z_test = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)

    with torch.no_grad():
        fake_test = generator(z_test)
        e_real_test = dem(sample_imgs)
        e_fake_test = dem(fake_test)

    print(f"\nForward pass check:")
    print(f"  Generator output shape   : {fake_test.shape}")
    print(f"  Energy (real) shape      : {e_real_test.shape}")
    print(f"  Energy (real) mean       : {e_real_test.mean().item():.4f}")
    print(f"  Energy (fake) mean       : {e_fake_test.mean().item():.4f}")
    print(f"  [OK] DDGM pipeline correct\n")

    # ── Training ─────────────────────────────────────────────────
    dem_losses = []
    gen_losses = []
    e_reals = []
    e_fakes = []
    best_gen_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"-- Epoch {epoch}/{EPOCHS} --------------------------")

        dem_loss, gen_loss, e_real, e_fake = train_one_epoch(
            generator, dem,
            optimizer_G, optimizer_DEM,
            train_loader, device,
            LATENT_DIM, D_STEPS,
            GRADIENT_PENALTY
        )

        dem_losses.append(dem_loss)
        gen_losses.append(gen_loss)
        e_reals.append(e_real)
        e_fakes.append(e_fake)

        print(f"  DEM_loss: {dem_loss:.4f}  Gen_loss: {gen_loss:.4f}  "
              f"E_real: {e_real:.2f}  E_fake: {e_fake:.2f}")

        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            torch.save(generator.state_dict(), 'ddgm_generator_best.pth')

        # Save progress every 10 epochs
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            fig, axes = plt.subplots(1, 8, figsize=(20, 3))
            for i, ax in enumerate(axes):
                ax.imshow(denorm(samples[i]))
                ax.axis('off')
            plt.suptitle(f"DDGM — Epoch {epoch} samples", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR,
                        f'ddgm_progress_epoch{epoch}.png'), dpi=100)
            plt.close()
            print(f"  Progress saved")

    # ── Save final models ────────────────────────────────────────
    torch.save(generator.state_dict(), 'ddgm_generator_final.pth')
    torch.save(dem.state_dict(), 'ddgm_dem_final.pth')
    print("\nModels saved.")

    # ── Final visualisations ─────────────────────────────────────
    plot_losses(dem_losses, gen_losses,
                save_path=os.path.join(RESULTS_DIR, 'ddgm_loss.png'))
    plot_energies(e_reals, e_fakes,
                  save_path=os.path.join(RESULTS_DIR, 'ddgm_energies.png'))
    visualise_random(generator, device, LATENT_DIM,
                     n=8, save_path=os.path.join(RESULTS_DIR,
                     'ddgm_random_generated.png'))

    print(f"\nAll results saved to {RESULTS_DIR}/")
