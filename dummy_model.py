import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import our dataset class
from dataset import GlassesDataset


# ─────────────────────────────────────────────
#  Dummy Autoencoder
#  Purpose: verify the pipeline works, NOT to generate good images
#  Architecture: simple conv encoder → conv decoder
# ─────────────────────────────────────────────

class DummyAutoencoder(nn.Module):
    """
    Tiny autoencoder with:
      Encoder: compress (3, 64, 64) → (32, 8, 8)
      Decoder: reconstruct (32, 8, 8) → (3, 64, 64)

    Hyperparameters are exposed as arguments so you can
    easily change them later (as the TA instructed).
    """

    def __init__(
        self,
        in_channels=3,        # RGB images
        base_filters=32,      # number of filters in first conv layer
        latent_channels=32,   # channels at the bottleneck
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        # (3, 64, 64) → (base_filters, 32, 32) → (base_filters*2, 16, 16) → (latent_channels, 8, 8)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(base_filters * 2, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────
        # (latent_channels, 8, 8) → (base_filters*2, 16, 16) → (base_filters, 32, 32) → (3, 64, 64)
        self.decoder = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(latent_channels, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            # Block 2
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),

            # Block 3 — output layer uses Tanh to match [-1, 1] normalisation
            nn.ConvTranspose2d(base_filters, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)          # move to GPU/CPU
        # labels not used here (unsupervised reconstruction)

        # ── Forward pass ─────────────────────────────────────────
        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        # ── Backward pass ────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ─────────────────────────────────────────────
#  Visualise reconstructions
# ─────────────────────────────────────────────

def visualise_reconstructions(model, dataset, device, n=6, save_path=None):
    """Show original vs reconstructed images side by side."""
    model.eval()
    indices = list(range(n))

    originals     = []
    reconstructed = []

    with torch.no_grad():
        for idx in indices:
            img_tensor, label = dataset[idx]
            inp = img_tensor.unsqueeze(0).to(device)   # (1, 3, 64, 64)
            out = model(inp).squeeze(0).cpu()           # (3, 64, 64)
            originals.append(img_tensor)
            reconstructed.append(out)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    for i in range(n):
        for row, imgs, title in zip([0, 1],
                                    [originals, reconstructed],
                                    ['Original', 'Reconstructed']):
            img = imgs[i].permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)   # denormalise [-1,1] → [0,1]
            axes[row, i].imshow(img)
            axes[row, i].axis('off')
            if i == 0:
                axes[row, i].set_title(title, fontsize=10)

    plt.suptitle("Dummy Autoencoder — Reconstruction Check", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Main — run this file to test the pipeline
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ── Config ───────────────────────────────────────────────────
    CSV_PATH   = 'final_train.csv'
    IMAGES_DIR = 'images'
    EPOCHS     = 1           # TA: use 1 epoch to verify pipeline first
    BATCH_SIZE = 32
    LR         = 1e-3
    TARGET_SIZE = (64, 64)

    # ── Device ───────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Datasets & Loaders ───────────────────────────────────────
    train_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='train', target_size=TARGET_SIZE, augment=True,
    )
    test_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='test', target_size=TARGET_SIZE, augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    # ── Model, Loss, Optimiser ───────────────────────────────────
    model     = DummyAutoencoder(in_channels=3, base_filters=32).to(device)
    criterion = nn.MSELoss()          # reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nModel parameter count: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Verify one forward pass BEFORE training ───────────────────
    # (catches shape errors immediately)
    sample_batch = next(iter(train_loader))
    sample_imgs  = sample_batch[0].to(device)
    with torch.no_grad():
        sample_out = model(sample_imgs)
    print(f"\nForward pass check:")
    print(f"  Input shape  : {sample_imgs.shape}")   # (32, 3, 64, 64)
    print(f"  Output shape : {sample_out.shape}")    # (32, 3, 64, 64)
    assert sample_imgs.shape == sample_out.shape, "Shape mismatch!"
    print("  ✅ Shapes match — pipeline is correct\n")

    # ── Training ─────────────────────────────────────────────────
    train_losses = []
    test_losses  = []

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} ──────────────────────────")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss  = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # ── Save model checkpoint ────────────────────────────────────
    torch.save(model.state_dict(), 'dummy_model.pth')
    print("\nModel saved to dummy_model.pth")

    # ── Verify model loads correctly ─────────────────────────────
    loaded_model = DummyAutoencoder(in_channels=3, base_filters=32).to(device)
    loaded_model.load_state_dict(torch.load('dummy_model.pth', map_location=device, weights_only=True))
    print("✅ Model loaded successfully")

    # ── Visualise reconstructions ────────────────────────────────
    visualise_reconstructions(
        model, test_dataset, device,
        n=6, save_path='reconstruction_check.png'
    )
