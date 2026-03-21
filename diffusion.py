import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from dataset import GlassesDataset


# ─────────────────────────────────────────────
#  Sinusoidal Timestep Embedding
#  Maps integer timestep → dense vector
# ─────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal position embedding (Vaswani et al.)
    adapted for diffusion timesteps.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)   # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)     # (B, dim)
        return emb


# ─────────────────────────────────────────────
#  Residual Block (with time + class embedding)
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU + residual
    Time/class embedding is projected and added after first norm.
    """

    def __init__(self, in_ch, out_ch, emb_dim, num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)

        # project embedding to channel dim
        self.emb_proj = nn.Linear(emb_dim, out_ch)

        # residual shortcut
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)

        # inject time + class embedding
        emb_out = self.emb_proj(F.silu(emb))            # (B, out_ch)
        h = h + emb_out[:, :, None, None]                # broadcast to spatial

        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.shortcut(x)


# ─────────────────────────────────────────────
#  Downsample / Upsample
# ─────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ─────────────────────────────────────────────
#  Conditional UNet  (noise-prediction network)
#  Input:  noisy image x_t (B, 3, 64, 64)
#          timestep t   (B,)
#          class label   (B,)
#  Output: predicted noise ε (B, 3, 64, 64)
# ─────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Hyperparameters (for ablation):
        base_dim        : base channel width (try 32, 64, 128)
        dim_mults       : channel multipliers at each level
        num_res_blocks  : residual blocks per level (try 1, 2, 3)
        num_classes     : number of class labels
        label_embed_dim : class embedding size
    """

    def __init__(
        self,
        in_channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        num_classes=2,
        label_embed_dim=128,
        num_groups=8,
    ):
        super().__init__()

        emb_dim = base_dim * 4      # internal embedding dimension

        # ── Time embedding ──────────────────────────────────────
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # ── Class embedding ─────────────────────────────────────
        self.class_embed = nn.Embedding(num_classes, emb_dim)

        # ── Initial conv ────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # ── Encoder (downsampling) ──────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_dim]
        ch = base_dim
        for i, mult in enumerate(dim_mults):
            out_ch = base_dim * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, emb_dim, num_groups))
                ch = out_ch
                channels.append(ch)
            if i < len(dim_mults) - 1:           # no downsample at last level
                self.down_samples.append(Downsample(ch))
                channels.append(ch)

        # ── Bottleneck ──────────────────────────────────────────
        self.mid_block1 = ResBlock(ch, ch, emb_dim, num_groups)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, num_groups)

        # ── Decoder (upsampling) ────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(dim_mults))):
            out_ch = base_dim * mult
            for _ in range(num_res_blocks + 1):  # +1 for skip connection
                skip_ch = channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, emb_dim, num_groups))
                ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(ch))

        # ── Final conv ──────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t, labels):
        # embeddings
        t_emb = self.time_embed(t)                       # (B, emb_dim)
        c_emb = self.class_embed(labels)                 # (B, emb_dim)
        emb   = t_emb + c_emb                            # additive conditioning

        # initial conv
        h = self.init_conv(x)
        skips = [h]

        # ── Encoder ─────────────────────────────────────────────
        down_idx = 0
        block_idx = 0
        for i, mult in enumerate(self._dim_mults()):
            for _ in range(self._num_res_blocks()):
                h = self.down_blocks[block_idx](h, emb)
                skips.append(h)
                block_idx += 1
            if i < len(self._dim_mults()) - 1:
                h = self.down_samples[down_idx](h)
                skips.append(h)
                down_idx += 1

        # ── Bottleneck ──────────────────────────────────────────
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # ── Decoder ─────────────────────────────────────────────
        up_idx = 0
        block_idx = 0
        for i, mult in reversed(list(enumerate(self._dim_mults()))):
            for _ in range(self._num_res_blocks() + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, emb)
                block_idx += 1
            if i > 0:
                h = self.up_samples[up_idx](h)
                up_idx += 1

        return self.final_conv(h)

    # helpers to access config (used in forward)
    def _dim_mults(self):
        # reconstruct from down_blocks structure
        # store them at init for easy access
        return self._dm

    def _num_res_blocks(self):
        return self._nrb


# ─────────────────────────────────────────────
#  Build UNet with stored config (cleaner API)
# ─────────────────────────────────────────────

def build_unet(
    in_channels=3,
    base_dim=64,
    dim_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    num_classes=2,
    label_embed_dim=128,
    num_groups=8,
):
    model = ConditionalUNet(
        in_channels=in_channels,
        base_dim=base_dim,
        dim_mults=dim_mults,
        num_res_blocks=num_res_blocks,
        num_classes=num_classes,
        label_embed_dim=label_embed_dim,
        num_groups=num_groups,
    )
    # store config for forward-pass helpers
    model._dm = dim_mults
    model._nrb = num_res_blocks
    return model


# ─────────────────────────────────────────────
#  Gaussian Diffusion (DDPM)
# ─────────────────────────────────────────────

class GaussianDiffusion:
    """
    DDPM forward + reverse process.

    Hyperparameters (for ablation):
        timesteps  : T — number of diffusion steps (try 200, 500, 1000)
        beta_start : β_1 (try 1e-5, 1e-4, 5e-4)
        beta_end   : β_T (try 0.01, 0.02, 0.05)
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps,
                                     dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance  σ²_t = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # ── Forward process (noising) ────────────────────────────
    def q_sample(self, x_start, t, noise=None):
        """
        q(x_t | x_0) = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    # ── Training loss ────────────────────────────────────────
    def p_losses(self, model, x_start, t, labels, noise=None):
        """
        Simple MSE noise-prediction loss (Ho et al. 2020).
        L = ‖ε − ε_θ(x_t, t)‖²
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, labels)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    # ── Reverse process (sampling) ───────────────────────────
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels):
        """
        Single reverse step: x_{t-1} from x_t.
        """
        beta_t     = self._extract(self.betas, t, x.shape)
        sqrt_one_m = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Predicted mean
        predicted_noise = model(x, t, labels)
        model_mean = sqrt_recip * (x - beta_t / sqrt_one_m * predicted_noise)

        if t_index == 0:
            return model_mean
        else:
            posterior_var = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels):
        """
        Full reverse process: x_T → x_0.
        """
        device = self.device
        b = shape[0]

        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, labels)

        return x

    @torch.no_grad()
    def generate(self, model, num_samples, label, device):
        """
        Generate images for a specific class.
        """
        model.eval()
        self.device = device
        labels = torch.full((num_samples,), label, dtype=torch.long, device=device)
        images = self.p_sample_loop(model, (num_samples, 3, 64, 64), labels)
        return images.clamp(-1, 1)

    # ── Utility ──────────────────────────────────────────────
    @staticmethod
    def _extract(tensor, t, shape):
        """Gather values at index t and reshape for broadcasting."""
        out = tensor.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, diffusion, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Sample random timesteps for each image in the batch
        t = torch.randint(0, diffusion.timesteps, (images.size(0),),
                          device=device, dtype=torch.long)

        loss = diffusion.p_losses(model, images, t, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * 0.5 + 0.5).clip(0, 1)


def visualise_class_generation(model, diffusion, device, save_path=None):
    """Generate 3 glasses + 3 no-glasses using conditional diffusion."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for row, (label, name) in enumerate([(1, 'Glasses'), (0, 'No Glasses')]):
        imgs = diffusion.generate(model, 3, label, device)
        for col in range(3):
            axes[row, col].imshow(denorm(imgs[col]))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(name, fontsize=10)

    plt.suptitle("DDPM — Class-Specific Generation", fontsize=12)
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
    plt.ylabel('MSE Loss')
    plt.title('DDPM Training Loss')
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

    # ── Config ───────────────────────────────────────────────
    CSV_PATH       = 'final_train.csv'
    IMAGES_DIR     = 'images'
    EPOCHS         = 50
    BATCH_SIZE     = 32          # ablation: try 16, 32, 64
    LR             = 2e-4        # ablation: try 1e-4, 2e-4, 5e-4
    TIMESTEPS      = 1000        # ablation: try 200, 500, 1000
    BETA_START     = 1e-4        # ablation: try 1e-5, 1e-4, 5e-4
    BETA_END       = 0.02        # ablation: try 0.01, 0.02, 0.05
    BASE_DIM       = 64          # ablation: try 32, 64, 128
    DIM_MULTS      = (1, 2, 4, 8)
    NUM_RES_BLOCKS = 2           # ablation: try 1, 2, 3

    # ── Device ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset — use ALL 5000 images ────────────────────────
    full_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='all', target_size=(64, 64), augment=True,
    )
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2,
                              drop_last=True)

    print(f"Training on {len(full_dataset)} images")

    # ── Model ────────────────────────────────────────────────
    model = build_unet(
        base_dim=BASE_DIM,
        dim_mults=DIM_MULTS,
        num_res_blocks=NUM_RES_BLOCKS,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Model parameter count: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Forward pass check ───────────────────────────────────
    sample_imgs, sample_labels = next(iter(train_loader))
    sample_imgs   = sample_imgs.to(device)
    sample_labels = sample_labels.to(device)
    t_test        = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=device)

    with torch.no_grad():
        noise_pred = model(sample_imgs, t_test, sample_labels)

    print(f"\nForward pass check:")
    print(f"  Input shape          : {sample_imgs.shape}")
    print(f"  Noise pred shape     : {noise_pred.shape}")
    assert sample_imgs.shape == noise_pred.shape, "Shape mismatch!"
    print(f"  ✅ Diffusion pipeline correct\n")

    # ── Training ─────────────────────────────────────────────
    train_losses = []
    best_loss    = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} ──────────────────────────")

        train_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device
        )
        train_losses.append(train_loss)
        scheduler.step()

        print(f"  Loss: {train_loss:.6f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'diffusion_best.pth')
            print(f"  💾 Best model saved (loss: {best_loss:.6f})")

        # Save progress every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # generate 4 glasses + 4 no-glasses
                g_labels  = torch.ones(4, dtype=torch.long, device=device)
                ng_labels = torch.zeros(4, dtype=torch.long, device=device)
                g_imgs    = diffusion.generate(model, 4, 1, device)
                ng_imgs   = diffusion.generate(model, 4, 0, device)
                samples   = torch.cat([g_imgs, ng_imgs], dim=0)

            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            titles = ['G', 'G', 'G', 'G', 'NG', 'NG', 'NG', 'NG']
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(denorm(samples[i]))
                ax.set_title(titles[i], fontsize=8)
                ax.axis('off')
            plt.suptitle(f"DDPM Epoch {epoch} — top: Glasses, bottom: No Glasses",
                         fontsize=10)
            plt.tight_layout()
            plt.savefig(f'diffusion_progress_epoch{epoch}.png', dpi=100)
            plt.close()
            print(f"  📸 Progress saved to diffusion_progress_epoch{epoch}.png")

    # ── Save final ───────────────────────────────────────────
    torch.save(model.state_dict(), 'diffusion_final.pth')
    print("\nFinal model saved to diffusion_final.pth")

    # ── Final visualisations ─────────────────────────────────
    plot_losses(train_losses, save_path='diffusion_loss.png')

    visualise_class_generation(
        model, diffusion, device,
        save_path='diffusion_class_generated.png'
    )
