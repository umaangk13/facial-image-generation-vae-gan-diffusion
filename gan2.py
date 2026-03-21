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
#  Conditional Generator (cGAN)
#  Input:  noise z + class label
#  Output: fake image conditioned on label
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    Hyperparameters (for ablation):
        latent_dim   : size of noise vector (try 64, 128, 256)
        base_filters : base filters (try 32, 64)
        activation   : 'relu' or 'leakyrelu'
        dropout      : dropout rate (try 0.0, 0.3, 0.5)
    """

    def __init__(
        self,
        latent_dim=128,
        base_filters=64,
        activation='relu',
        dropout=0.0,
        num_classes=2,
        label_embed_dim=32,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.base_filters = base_filters

        def act():
            return (nn.ReLU(inplace=True)
                    if activation == 'relu'
                    else nn.LeakyReLU(0.2, inplace=True))

        def drop():
            return nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Label embedding: class → dense vector
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)

        # Project (z + label_embed) → spatial feature map (8f, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_embed_dim, base_filters * 8 * 4 * 4),
            nn.BatchNorm1d(base_filters * 8 * 4 * 4),
            act(),
        )

        # Upsample: (8f,4,4)→(4f,8,8)→(2f,16,16)→(f,32,32)→(3,64,64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 4, 2, 1),
            nn.BatchNorm2d(base_filters * 4),
            act(),
            drop(),

            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, 2, 1),
            nn.BatchNorm2d(base_filters * 2),
            act(),
            drop(),

            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1),
            nn.BatchNorm2d(base_filters),
            act(),
            drop(),

            nn.ConvTranspose2d(base_filters, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        le = self.label_embed(labels)       # (B, embed_dim)
        x  = torch.cat([z, le], dim=1)     # (B, latent_dim + embed_dim)
        x  = self.fc(x)
        x  = x.view(x.size(0), self.base_filters * 8, 4, 4)
        return self.deconv(x)


# ─────────────────────────────────────────────
#  Conditional Discriminator (cGAN)
#  Input:  image + class label
#  Output: real/fake probability
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Hyperparameters (for ablation):
        base_filters : base filters (try 32, 64)
        dropout      : dropout rate (try 0.0, 0.3, 0.5)
        leaky_slope  : LeakyReLU slope (try 0.1, 0.2, 0.3)
    """

    def __init__(
        self,
        base_filters=64,
        dropout=0.3,
        leaky_slope=0.2,
        num_classes=2,
    ):
        super().__init__()

        def lrelu():
            return nn.LeakyReLU(leaky_slope, inplace=True)

        def drop():
            return nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Label embedding: project to spatial map (1, 64, 64)
        # and concatenate with image as extra channel
        self.label_embed = nn.Embedding(num_classes, 64 * 64)

        # Input: image (3 ch) + label map (1 ch) = 4 channels
        self.conv = nn.Sequential(
            # No BatchNorm in first layer — GAN hack
            nn.Conv2d(4, base_filters, 4, 2, 1),
            lrelu(),
            drop(),

            nn.Conv2d(base_filters,     base_filters * 2, 4, 2, 1),
            nn.BatchNorm2d(base_filters * 2),
            lrelu(),
            drop(),

            nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),
            nn.BatchNorm2d(base_filters * 4),
            lrelu(),
            drop(),

            nn.Conv2d(base_filters * 4, base_filters * 8, 4, 2, 1),
            nn.BatchNorm2d(base_filters * 8),
            lrelu(),
            drop(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        # Embed label → spatial map (B, 1, 64, 64)
        le = self.label_embed(labels)
        le = le.view(-1, 1, 64, 64)
        x  = torch.cat([x, le], dim=1)     # (B, 4, 64, 64)
        x  = self.conv(x)
        return self.fc(x)


# ─────────────────────────────────────────────
#  Training Steps
# ─────────────────────────────────────────────

def train_discriminator(discriminator, generator, real_images, real_labels,
                         optimizer_D, criterion, device,
                         latent_dim, label_smoothing=0.1):
    """
    Ablation: label_smoothing (try 0.0, 0.1, 0.2)
    """
    discriminator.train()
    batch_size = real_images.size(0)

    real_targets = torch.full((batch_size, 1),
                               1.0 - label_smoothing, device=device)
    fake_targets = torch.zeros(batch_size, 1, device=device)

    # ── Real images with their true labels ───────────────────────
    optimizer_D.zero_grad()
    real_output = discriminator(real_images, real_labels)
    loss_real   = criterion(real_output, real_targets)
    loss_real.backward()

    # ── Fake images with random labels ───────────────────────────
    z            = torch.randn(batch_size, latent_dim, device=device)
    fake_labels  = torch.randint(0, 2, (batch_size,), device=device)
    fake_images  = generator(z, fake_labels).detach()
    fake_output  = discriminator(fake_images, fake_labels)
    loss_fake    = criterion(fake_output, fake_targets)
    loss_fake.backward()

    optimizer_D.step()

    d_loss = (loss_real + loss_fake).item()
    d_acc  = ((real_output > 0.5).float().mean().item() +
               (fake_output < 0.5).float().mean().item()) / 2
    return d_loss, d_acc


def train_generator(generator, discriminator, optimizer_G,
                     criterion, batch_size, latent_dim, device):
    generator.train()

    # Generator wants discriminator to think fakes are real
    # Use random labels so generator learns both classes
    real_targets = torch.ones(batch_size, 1, device=device)
    fake_labels  = torch.randint(0, 2, (batch_size,), device=device)

    optimizer_G.zero_grad()
    z           = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z, fake_labels)
    fake_output = discriminator(fake_images, fake_labels)
    g_loss      = criterion(fake_output, real_targets)
    g_loss.backward()
    optimizer_G.step()

    return g_loss.item()


def train_one_epoch(generator, discriminator,
                     optimizer_G, optimizer_D,
                     dataloader, criterion, device,
                     latent_dim, d_steps, label_smoothing):
    """
    Ablation: d_steps (try 1, 2, 3)
    """
    total_d_loss = total_g_loss = total_d_acc = 0.0
    n_batches = 0

    for batch_idx, (real_images, real_labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size  = real_images.size(0)

        for _ in range(d_steps):
            d_loss, d_acc = train_discriminator(
                discriminator, generator,
                real_images, real_labels,
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


def visualise_class_generation(generator, device, latent_dim,
                                 save_path=None):
    """
    Proper conditional generation:
    Pass label=1 to generator → glasses faces
    Pass label=0 to generator → no glasses faces
    """
    generator.eval()
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    with torch.no_grad():
        for row, (label, name) in enumerate([(1, 'Glasses'),
                                              (0, 'No Glasses')]):
            z      = torch.randn(3, latent_dim, device=device)
            labels = torch.full((3,), label, dtype=torch.long, device=device)
            imgs   = generator(z, labels)

            for col in range(3):
                axes[row, col].imshow(denorm(imgs[col]))
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_title(name, fontsize=10)

    plt.suptitle("cGAN — Class-Specific Generation", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def visualise_random(generator, device, latent_dim, n=8, save_path=None):
    generator.eval()
    with torch.no_grad():
        z      = torch.randn(n, latent_dim, device=device)
        labels = torch.randint(0, 2, (n,), device=device)
        imgs   = generator(z, labels)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, ax in enumerate(axes):
        ax.imshow(denorm(imgs[i]))
        ax.set_title("G" if labels[i].item() == 1 else "NG", fontsize=8)
        ax.axis('off')

    plt.suptitle("cGAN — Random Generation", fontsize=12)
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
    plt.title('cGAN Training Loss')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Weight Initialisation (DCGAN standard)
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
    LR_G            = 2e-4      # ablation: try 1e-4, 2e-4
    LR_D            = 2e-4
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

    # ── Optimisers — beta1=0.5 is standard GAN hack ──────────────
    optimizer_G = optim.Adam(generator.parameters(),
                              lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                              lr=LR_D, betas=(0.5, 0.999))
    criterion   = nn.BCELoss()

    # Fixed noise for tracking progress (same z every epoch)
    fixed_noise  = torch.randn(8, LATENT_DIM, device=device)
    fixed_labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0],
                                  dtype=torch.long, device=device)

    # ── Forward pass check ───────────────────────────────────────
    sample_imgs, sample_labels = next(iter(train_loader))
    sample_imgs   = sample_imgs.to(device)
    sample_labels = sample_labels.to(device)
    z_test        = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    l_test        = torch.randint(0, 2, (BATCH_SIZE,), device=device)

    with torch.no_grad():
        fake_test  = generator(z_test, l_test)
        d_real     = discriminator(sample_imgs, sample_labels)
        d_fake     = discriminator(fake_test, l_test)

    print(f"\nForward pass check:")
    print(f"  Generator output : {fake_test.shape}")
    print(f"  D(real) output   : {d_real.shape}")
    print(f"  D(fake) output   : {d_fake.shape}")
    print(f"  ✅ cGAN pipeline correct\n")

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
            torch.save(generator.state_dict(), 'gan_generator_best.pth')

        # Save progress images every 10 epochs
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise, fixed_labels)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            titles = ['G', 'G', 'G', 'G', 'NG', 'NG', 'NG', 'NG']
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(denorm(samples[i]))
                ax.set_title(titles[i], fontsize=8)
                ax.axis('off')
            plt.suptitle(f"Epoch {epoch} — top: Glasses, bottom: No Glasses",
                         fontsize=10)
            plt.tight_layout()
            plt.savefig(f'gan_progress_epoch{epoch}.png', dpi=100)
            plt.close()
            print(f"  📸 Progress saved to gan_progress_epoch{epoch}.png")

    # ── Save final ───────────────────────────────────────────────
    torch.save(generator.state_dict(),     'gan_generator_final.pth')
    torch.save(discriminator.state_dict(), 'gan_discriminator_final.pth')
    print("\nModels saved.")

    # ── Final visualisations ─────────────────────────────────────
    plot_losses(d_losses, g_losses, save_path='gan_loss.png')

    visualise_random(
        generator, device, LATENT_DIM,
        n=8, save_path='gan_random_generated.png'
    )

    visualise_class_generation(
        generator, device, LATENT_DIM,
        save_path='gan_class_generated.png'
    )
