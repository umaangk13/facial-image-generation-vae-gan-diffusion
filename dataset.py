import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Suppress albumentations update warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


# ─────────────────────────────────────────────
#  Dataset Class
# ─────────────────────────────────────────────

class GlassesDataset(Dataset):
    """
    Loads face images from disk, resizes to target_size,
    applies optional Albumentations augmentations,
    and returns (image_tensor, label) pairs.

    Args:
        csv_path      : path to your final_train.csv
        images_dir    : folder where images are stored
        split         : 'train', 'test', or 'all'
                        Use 'all' for generative models (VAE, GAN, Diffusion)
                        Use 'train'/'test' only for classifiers
        test_fraction : fraction of data to hold out as test (default 0.2)
                        Ignored when split='all'
        target_size   : (H, W) to resize images to (default 64x64)
        augment       : whether to apply augmentations
        seed          : random seed for reproducible train/test split
    """

    def __init__(
        self,
        csv_path,
        images_dir,
        split='all',
        test_fraction=0.2,
        target_size=(64, 64),
        augment=True,
        seed=42,
    ):
        self.images_dir  = images_dir
        self.target_size = target_size
        self.split       = split

        # ── Load CSV ──────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        df = df[['id', 'glasses']].copy()
        df['glasses'] = df['glasses'].astype(int)

        # ── Split logic ───────────────────────────────────────────
        if split == 'all':
            # Use ALL 5000 images — correct for generative models
            self.df = df.reset_index(drop=True)
        else:
            # Train/test split — only needed for classifiers
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            n_test  = int(len(df) * test_fraction)
            n_train = len(df) - n_test

            if split == 'train':
                self.df = df.iloc[:n_train].reset_index(drop=True)
            elif split == 'test':
                self.df = df.iloc[n_train:].reset_index(drop=True)
            else:
                raise ValueError(f"split must be 'all', 'train' or 'test', "
                                 f"got '{split}'")

        print(f"[{split}] {len(self.df)} samples "
              f"| glasses=1: {self.df['glasses'].sum()} "
              f"| glasses=0: {(self.df['glasses']==0).sum()}")

        # ── Augmentation pipeline ─────────────────────────────────
        if augment and split != 'test':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.4
                ),
                A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])

    # ── Helpers ───────────────────────────────────────────────────

    def _find_image_path(self, image_id):
        candidates = [
            f"face-{image_id}.png",
            f"face-{image_id}.jpg",
            f"{image_id}.png",
            f"{image_id}.jpg",
        ]
        for name in candidates:
            path = os.path.join(self.images_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No image found for id={image_id} in {self.images_dir}"
        )

    # ── Core methods ──────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = int(row['id'])
        label    = int(row['glasses'])

        img_path = self._find_image_path(image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"cv2 could not read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        augmented  = self.transform(image=img)
        img_tensor = augmented['image'].float()

        return img_tensor, label

    # ── Utility ───────────────────────────────────────────────────

    def visualise_samples(self, n=8, save_path=None):
        indices = np.random.choice(len(self), size=min(n, len(self)), replace=False)
        fig, axes = plt.subplots(1, len(indices), figsize=(2.5 * len(indices), 3))
        if len(indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            img_tensor, label = self[idx]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            ax.imshow(img_np)
            ax.set_title("Glasses" if label == 1 else "No Glasses", fontsize=9)
            ax.axis('off')

        plt.suptitle(f"Split: {self.split} | size: {self.target_size}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == '__main__':

    CSV_PATH   = 'final_train.csv'
    IMAGES_DIR = 'images'

    # For generative models — use all data
    full_dataset = GlassesDataset(
        csv_path=CSV_PATH, images_dir=IMAGES_DIR,
        split='all', augment=True,
    )

    img, label = full_dataset[0]
    print(f"\nSample check:")
    print(f"  Image tensor shape : {img.shape}")
    print(f"  Pixel value range  : [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Label              : {label}")

    loader = DataLoader(full_dataset, batch_size=32,
                        shuffle=True, num_workers=2)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"\nBatch check:")
    print(f"  Batch image shape : {batch_imgs.shape}")
    print(f"  Batch label shape : {batch_labels.shape}")

    full_dataset.visualise_samples(n=8, save_path='sample_check.png')
