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
#  Residual Block
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Standard ResNet-style block:
    x' = x + Conv(x)
    If input_channels != output_channels or spatial size changes (stride > 1),
    a 1x1 conv shortcut is applied to x.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        
        act = {
            'relu':      nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.2, inplace=True),
            'elu':       nn.ELU(inplace=True),
        }[activation]
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            act
        )
        
        # Shortcut connection
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResBlockTranspose(nn.Module):
    """Residual block with ConvTranspose for decoding / upsampling"""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, activation='relu'):
        super().__init__()
        
        act = {
            'relu':      nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.2, inplace=True),
            'elu':       nn.ELU(inplace=True),
        }[activation]
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            act
        )
        
        # Shortcut using normal Conv2d with upsampling, or ConvTranspose
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.deconv(x) + self.shortcut(x)


class ResBlockUpsample(nn.Module):
    """Residual block with bilinear upsampling + Conv2d"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        
        act = {
            'relu':      nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.2, inplace=True),
            'elu':       nn.ELU(inplace=True),
        }[activation]
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            act
        )
        
        if in_ch != out_ch:
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 1, 0),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut_conv = nn.Identity()
            
    def forward(self, x):
        up = self.upsample(x)
        return self.conv(up) + self.shortcut_conv(up)


# ─────────────────────────────────────────────
#  Conditional VAE (CVAE)
# ─────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """
    Encoder: (image + label) → (mu, log_var)
    Uses residual connections.
    """

    def __init__(
        self,
        in_channels=3,
        base_filters=32,
        latent_dim=128,
        conv_kernel=3,
        pool_stride=2,
        activation='relu',
        num_classes=2,
        label_embed_dim=32,
    ):
        super().__init__()

        pad = conv_kernel // 2

        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.label_proj  = nn.Linear(label_embed_dim, 64 * 64)

        self.conv = nn.Sequential(
            ResBlock(in_channels + 1, base_filters, conv_kernel, pool_stride, pad, activation),
            ResBlock(base_filters, base_filters * 2, conv_kernel, pool_stride, pad, activation),
            ResBlock(base_filters * 2, base_filters * 4, conv_kernel, pool_stride, pad, activation),
            ResBlock(base_filters * 4, base_filters * 8, conv_kernel, pool_stride, pad, activation),
        )

        self.flat_dim   = base_filters * 8 * 4 * 4
        self.fc_mu      = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x, labels):
        le  = self.label_embed(labels)
        lp  = self.label_proj(le)
        lp  = lp.view(-1, 1, 64, 64)

        x   = torch.cat([x, lp], dim=1)
        x   = self.conv(x)
        x   = x.view(x.size(0), -1)
        mu      = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Decoder: (z + label) → image
    Uses residual connections.
    """

    def __init__(
        self,
        out_channels=3,
        base_filters=32,
        latent_dim=128,
        conv_kernel=3,
        activation='relu',
        use_interpolation=False,
        num_classes=2,
        label_embed_dim=32,
    ):
        super().__init__()

        self.base_filters = base_filters

        act = {
            'relu':      nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.2, inplace=True),
            'elu':       nn.ELU(inplace=True),
        }[activation]

        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.fc = nn.Linear(latent_dim + label_embed_dim, base_filters * 8 * 4 * 4)
        self.act = act

        pad = conv_kernel // 2

        if not use_interpolation:
            self.deconv = nn.Sequential(
                ResBlockTranspose(base_filters * 8, base_filters * 4, 4, 2, 1, activation),
                ResBlockTranspose(base_filters * 4, base_filters * 2, 4, 2, 1, activation),
                ResBlockTranspose(base_filters * 2, base_filters, 4, 2, 1, activation),
                nn.ConvTranspose2d(base_filters, out_channels, 4, 2, 1),
                nn.Tanh(),
            )
        else:
            self.deconv = nn.Sequential(
                ResBlockUpsample(base_filters * 8, base_filters * 4, conv_kernel, 1, pad, activation),
                ResBlockUpsample(base_filters * 4, base_filters * 2, conv_kernel, 1, pad, activation),
                ResBlockUpsample(base_filters * 2, base_filters, conv_kernel, 1, pad, activation),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(base_filters, out_channels, conv_kernel, 1, pad),
                nn.Tanh(),
            )

    def forward(self, z, labels):
        le = self.label_embed(labels)           # (B, embed_dim)
        z  = torch.cat([z, le], dim=1)          # (B, latent_dim + embed_dim)
        x  = self.fc(z)
        x  = self.act(x)
        x  = x.view(x.size(0), self.base_filters * 8, 4, 4)
        return self.deconv(x)


class CVAE(nn.Module):
    """
    Conditional VAE — label is fed into both encoder and decoder.
    This is what enables proper class-specific generation.

    Hyperparameters (for ablation):
        latent_dim        : size of z (try 64, 128, 256, 512)
        base_filters      : base conv filters (try 32, 64)
        conv_kernel       : filter size (try 3, 5, 7)
        pool_stride       : downsampling stride (try 2, 4)
        activation        : 'relu', 'leakyrelu', 'elu'
        use_interpolation : True = bilinear, False = deconv
        kl_weight         : weight on KL term (try 0.1, 0.5, 1.0, 2.0)
    """

    def __init__(
        self,
        latent_dim=128,
        base_filters=32,
        conv_kernel=3,
        pool_stride=2,
        activation='relu',
        use_interpolation=False,
        num_classes=2,
        label_embed_dim=32,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(
            in_channels=3,
            base_filters=base_filters,
            latent_dim=latent_dim,
            conv_kernel=conv_kernel,
            pool_stride=pool_stride,
            activation=activation,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
        )
        self.decoder = VAEDecoder(
            out_channels=3,
            base_filters=base_filters,
            latent_dim=latent_dim,
            conv_kernel=conv_kernel,
            activation=activation,
            use_interpolation=use_interpolation,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
        )

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, log_var  = self.encoder(x, labels)
        z            = self.reparameterise(mu, log_var)
        reconstructed = self.decoder(z, labels)
        return reconstructed, mu, log_var

    def generate(self, num_samples, label, device):
        """
        Generate images for a specific class by sampling z ~ N(0,1)
        and conditioning the decoder on the given label.
        This is proper conditional generation — not a bias trick.
        """
        self.eval()
        with torch.no_grad():
            z      = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), label,
                                dtype=torch.long, device=device)
            images = self.decoder(z, labels)
        return images


# ─────────────────────────────────────────────
#  VAE Loss
# ─────────────────────────────────────────────

def vae_loss(reconstructed, original, mu, log_var, kl_weight=0.5):
    recon_loss = nn.functional.mse_loss(reconstructed, original,
                                         reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        reconstructed, mu, log_var = model(images, labels)
        loss, recon_loss, kl_loss  = vae_loss(reconstructed, images,
                                               mu, log_var, kl_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"Loss: {loss.item():.4f}  "
                  f"Recon: {recon_loss.item():.4f}  "
                  f"KL: {kl_loss.item():.4f}")

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


def visualise_reconstructions(model, dataset, device, n=6, save_path=None):
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))

    with torch.no_grad():
        for i in range(n):
            img_tensor, label = dataset[i]
            inp    = img_tensor.unsqueeze(0).to(device)
            lbl    = torch.tensor([label], device=device)
            out, _, _ = model(inp, lbl)

            axes[0, i].imshow(denorm(img_tensor))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(denorm(out.squeeze(0)))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.suptitle("CVAE — Reconstructions", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def visualise_generated(model, device, save_path=None):
    """Generate 3 glasses and 3 no-glasses using proper conditional generation."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for row, (label, name) in enumerate([(1, 'Glasses'), (0, 'No Glasses')]):
        imgs = model.generate(3, label, device)
        for col in range(3):
            axes[row, col].imshow(denorm(imgs[col]))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(name, fontsize=10)

    plt.suptitle("CVAE — Class-Specific Generation", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def plot_losses(train_losses, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CVAE Training Loss')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── Config ───────────────────────────────────────────────────
    CSV_PATH          = 'final_train.csv'
    IMAGES_DIR        = 'images'
    EPOCHS            = 30
    BATCH_SIZE        = 32
    LR                = 1e-3
    KL_WEIGHT         = 0.5       # ablation: try 0.1, 0.5, 1.0, 2.0
    LATENT_DIM        = 128       # ablation: try 64, 128, 256, 512
    BASE_FILTERS      = 32        # ablation: try 32, 64
    CONV_KERNEL       = 3         # ablation: try 3, 5, 7
    ACTIVATION        = 'relu'    # ablation: try 'relu', 'leakyrelu', 'elu'
    USE_INTERP        = False      # ablation: try True vs False

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

    # ── Model ────────────────────────────────────────────────────
    model = CVAE(
        latent_dim=LATENT_DIM,
        base_filters=BASE_FILTERS,
        conv_kernel=CONV_KERNEL,
        activation=ACTIVATION,
        use_interpolation=USE_INTERP,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    print(f"Model parameter count: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Forward pass check ───────────────────────────────────────
    sample_imgs, sample_labels = next(iter(train_loader))
    sample_imgs   = sample_imgs.to(device)
    sample_labels = sample_labels.to(device)

    with torch.no_grad():
        out, mu, log_var = model(sample_imgs, sample_labels)

    print(f"\nForward pass check:")
    print(f"  Input shape  : {sample_imgs.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Mu shape     : {mu.shape}")
    print(f"  ✅ CVAE pipeline correct\n")

    # ── Training ─────────────────────────────────────────────────
    train_losses = []
    best_loss    = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} ──────────────────────────")

        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, KL_WEIGHT
        )
        train_losses.append(train_loss)
        scheduler.step(train_loss)

        print(f"  Total: {train_loss:.4f}  "
              f"Recon: {train_recon:.4f}  "
              f"KL: {train_kl:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'vae_best.pth')
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")

    # ── Save final ───────────────────────────────────────────────
    torch.save(model.state_dict(), 'vae_final.pth')
    print("\nFinal model saved to vae_final.pth")

    # ── Visualise ────────────────────────────────────────────────
    plot_losses(train_losses, save_path='vae_loss.png')

    visualise_reconstructions(
        model, full_dataset, device,
        n=6, save_path='vae_reconstructions.png'
    )

    visualise_generated(
        model, device,
        save_path='vae_generated.png'
    )
