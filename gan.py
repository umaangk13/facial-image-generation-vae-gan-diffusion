import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset


# ─────────────────────────────────────────────
#  Residual Blocks
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block for Discriminator."""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, leaky_slope=0.2, dropout=0.3, use_bn=True):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.conv = nn.Sequential(*layers)
        
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResBlockTranspose(nn.Module):
    """Residual block for Generator."""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, activation='relu', dropout=0.0):
        super().__init__()
        
        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            act
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.deconv = nn.Sequential(*layers)
        
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.deconv(x) + self.shortcut(x)


# ─────────────────────────────────────────────
#  Generator
#  Input:  random noise vector z (latent_dim,)
#  Output: fake image (3, 64, 64)
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    Uses residual connections.
    Hyperparameters (for ablation):
        latent_dim   : size of input noise vector (try 64, 128, 256)
        base_filters : base number of filters (try 32, 64)
        activation   : 'relu' or 'leakyrelu'
        dropout      : dropout rate (try 0.0, 0.3, 0.5)
    """

    def __init__(
        self,
        latent_dim=128,
        base_filters=64,
        activation='relu',
        dropout=0.0,
    ):
        super().__init__()

        self.latent_dim   = latent_dim
        self.base_filters = base_filters

        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)

        # Project noise → spatial feature map (8f, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_filters * 8 * 4 * 4),
            nn.BatchNorm1d(base_filters * 8 * 4 * 4),
            act,
        )

        # Upsample: (8f,4,4)→(4f,8,8)→(2f,16,16)→(f,32,32)→(3,64,64)
        self.deconv = nn.Sequential(
            ResBlockTranspose(base_filters * 8, base_filters * 4, 4, 2, 1, activation, dropout),
            ResBlockTranspose(base_filters * 4, base_filters * 2, 4, 2, 1, activation, dropout),
            ResBlockTranspose(base_filters * 2, base_filters, 4, 2, 1, activation, dropout),
            nn.ConvTranspose2d(base_filters, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.base_filters * 8, 4, 4)
        return self.deconv(x)


# ─────────────────────────────────────────────
#  Discriminator
#  Input:  image (3, 64, 64)
#  Output: scalar probability (real=1, fake=0)
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Uses residual connections.
    Hyperparameters (for ablation):
        base_filters : base number of filters (try 32, 64)
        dropout      : dropout rate (try 0.0, 0.3, 0.5)
        leaky_slope  : LeakyReLU negative slope (try 0.1, 0.2, 0.3)
    """

    def __init__(
        self,
        base_filters=64,
        dropout=0.3,
        leaky_slope=0.2,
    ):
        super().__init__()

        # (3,64,64)→(f,32,32)→(2f,16,16)→(4f,8,8)→(8f,4,4)
        self.conv = nn.Sequential(
            # No BatchNorm in first layer — GAN hack
            ResBlock(3, base_filters, 4, 2, 1, leaky_slope, dropout, use_bn=False),
            ResBlock(base_filters, base_filters * 2, 4, 2, 1, leaky_slope, dropout),
            ResBlock(base_filters * 2, base_filters * 4, 4, 2, 1, leaky_slope, dropout),
            ResBlock(base_filters * 4, base_filters * 8, 4, 2, 1, leaky_slope, dropout),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# ─────────────────────────────────────────────
#  Training Steps
# ─────────────────────────────────────────────

def train_discriminator(discriminator, generator, real_images,
                         optimizer_D, criterion, device,
                         latent_dim, label_smoothing=0.1):
    discriminator.train()
    batch_size = real_images.size(0)

    real_labels = torch.full((batch_size, 1),
                              1.0 - label_smoothing, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # ── Train on real ─────────────────────────────────────────────
    optimizer_D.zero_grad()
    real_output = discriminator(real_images)
    loss_real   = criterion(real_output, real_labels)
    loss_real.backward()

    # ── Train on fake ─────────────────────────────────────────────
    z           = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z).detach()
    fake_output = discriminator(fake_images)
    loss_fake   = criterion(fake_output, fake_labels)
    loss_fake.backward()

    optimizer_D.step()

    d_loss = (loss_real + loss_fake).item()
    d_acc  = ((real_output > 0.5).float().mean().item() +
               (fake_output < 0.5).float().mean().item()) / 2
    return d_loss, d_acc


def train_generator(generator, discriminator, optimizer_G,
                     criterion, batch_size, latent_dim, device):
    generator.train()

    real_targets = torch.ones(batch_size, 1, device=device)

    optimizer_G.zero_grad()
    z           = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    fake_output = discriminator(fake_images)
    g_loss      = criterion(fake_output, real_targets)
    g_loss.backward()
    optimizer_G.step()

    return g_loss.item()


def train_one_epoch(generator, discriminator,
                     optimizer_G, optimizer_D,
                     dataloader, criterion, device,
                     latent_dim, d_steps, label_smoothing):
    total_d_loss = total_g_loss = total_d_acc = 0.0
    n_batches = 0

    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size  = real_images.size(0)

        for _ in range(d_steps):
            d_loss, d_acc = train_discriminator(
                discriminator, generator, real_images,
                optimizer_D, criterion, device,
                latent_dim, label_smoothing
            )

        g_loss = train_generator(
            generator, discriminator,
            optimizer_G, criterion,
            batch_size, latent_dim, device
        )

        total_d_loss += d_loss
        total_g_loss += g_loss
        total_d_acc  += d_acc
        n_batches    += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"D_loss: {d_loss:.4f}  G_loss: {g_loss:.4f}  "
                  f"D_acc: {d_acc:.2f}")

    return (total_d_loss / n_batches,
            total_g_loss / n_batches,
            total_d_acc  / n_batches)


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


def visualise_random(generator, device, latent_dim, n=8, save_path=None):
    generator.eval()
    with torch.no_grad():
        z    = torch.randn(n, latent_dim, device=device)
        imgs = generator(z)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, ax in enumerate(axes):
        ax.imshow(denorm(imgs[i]))
        ax.axis('off')

    plt.suptitle("GAN — Random Generation", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def plot_losses(d_losses, g_losses, save_path=None):
    epochs = range(1, len(d_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Weight Initialisation
# ─────────────────────────────────────────────

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── Config ───────────────────────────────────────────────────
    CSV_PATH        = 'final_train.csv'
    IMAGES_DIR      = 'images'
    EPOCHS          = 50
    BATCH_SIZE      = 32
    LATENT_DIM      = 128       # ablation: try 64, 128, 256
    BASE_FILTERS    = 64        # ablation: try 32, 64
    LR_G            = 2e-4
    LR_D            = 2e-4      # ablation: try 1e-4, 2e-4
    D_STEPS         = 1         # ablation: try 1, 2, 3
    LABEL_SMOOTHING = 0.1       # ablation: try 0.0, 0.1, 0.2
    DROPOUT         = 0.3       # ablation: try 0.0, 0.3, 0.5
    G_ACTIVATION    = 'relu'    # ablation: try 'relu', 'leakyrelu'

    # ── Device ───────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset — use ALL 5000 images ────────────────────────────
    full_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='all', target_size=(64, 64), augment=True,
    )
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)

    print(f"Training on {len(full_dataset)} images")

    # ── Models ───────────────────────────────────────────────────
    generator = Generator(
        latent_dim=LATENT_DIM,
        base_filters=BASE_FILTERS,
        activation=G_ACTIVATION,
        dropout=DROPOUT,
    ).to(device)

    discriminator = Discriminator(
        base_filters=BASE_FILTERS,
        dropout=DROPOUT,
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(f"Generator params     : "
          f"{sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params : "
          f"{sum(p.numel() for p in discriminator.parameters()):,}")

    # ── Optimisers ───────────────────────────────────────────────
    optimizer_G = optim.Adam(generator.parameters(),
                              lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                              lr=LR_D, betas=(0.5, 0.999))
    criterion   = nn.BCELoss()

    # Fixed noise to track progress across epochs
    fixed_noise = torch.randn(8, LATENT_DIM, device=device)

    # ── Forward pass check ───────────────────────────────────────
    sample_imgs, _ = next(iter(train_loader))
    sample_imgs    = sample_imgs.to(device)
    z_test         = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)

    with torch.no_grad():
        fake_test  = generator(z_test)
        d_out_real = discriminator(sample_imgs)
        d_out_fake = discriminator(fake_test)

    print(f"\nForward pass check:")
    print(f"  Generator output shape      : {fake_test.shape}")
    print(f"  Discriminator (real) output : {d_out_real.shape}")
    print(f"  Discriminator (fake) output : {d_out_fake.shape}")
    print(f"  ✅ GAN pipeline correct\n")

    # ── Training ─────────────────────────────────────────────────
    d_losses    = []
    g_losses    = []
    best_g_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} ──────────────────────────")

        d_loss, g_loss, d_acc = train_one_epoch(
            generator, discriminator,
            optimizer_G, optimizer_D,
            train_loader, criterion, device,
            LATENT_DIM, D_STEPS, LABEL_SMOOTHING
        )

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        print(f"  D_loss: {d_loss:.4f}  G_loss: {g_loss:.4f}  "
              f"D_acc: {d_acc:.2f}")

        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save(generator.state_dict(), 'gan_unc_generator_best.pth')

        # Save progress every 10 epochs
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            fig, axes = plt.subplots(1, 8, figsize=(20, 3))
            for i, ax in enumerate(axes):
                ax.imshow(denorm(samples[i]))
                ax.axis('off')
            plt.suptitle(f"Epoch {epoch} samples", fontsize=11)
            plt.tight_layout()
            plt.savefig(f'gan_unc_progress_epoch{epoch}.png', dpi=100)
            plt.close()
            print(f"  📸 Progress saved to gan_unc_progress_epoch{epoch}.png")

    # ── Save final models ────────────────────────────────────────
    torch.save(generator.state_dict(),     'gan_unc_generator_final.pth')
    torch.save(discriminator.state_dict(), 'gan_unc_discriminator_final.pth')
    print("\nModels saved.")

    # ── Final visualisations ─────────────────────────────────────
    plot_losses(d_losses, g_losses, save_path='gan_unc_loss.png')
    visualise_random(
        generator, device, LATENT_DIM,
        n=8, save_path='gan_unc_random_generated.png'
    )