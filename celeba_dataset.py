"""
CelebA Dataset Wrapper for DDGM Training
==========================================
Loads CelebA face images (from torchvision or manual download)
and provides the same interface as GlassesDataset.

CelebA has ~200K aligned face images at 178×218, which we resize
to 64×64. The 'Eyeglasses' attribute (index 15) maps directly to
the 'glasses' label used in the original assignment dataset.

Setup options:
  1. Auto download via torchvision (may hit Google Drive limits):
       python celeba_dataset.py --download

  2. Manual download (recommended):
       - Download img_align_celeba.zip from:
         https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
         OR https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
       - Extract to: data/celeba/img_align_celeba/
       - Also place list_attr_celeba.txt in data/celeba/
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


class CelebAFaceDataset(Dataset):
    """
    CelebA dataset for face generation.
    
    Provides the same (image_tensor, label) interface as GlassesDataset.
    Label = 1 for glasses, 0 for no glasses (CelebA attribute #15).
    
    Args:
        root           : root dir containing celeba/ folder
        split          : 'train', 'valid', 'test', or 'all'
        target_size    : (H, W) to resize to
        augment        : apply data augmentations
        max_samples    : limit dataset size (None = use all)
        use_torchvision: if True, use torchvision.datasets.CelebA loader
                         if False, load from raw image folder
    """

    def __init__(
        self,
        root='data',
        split='train',
        target_size=(64, 64),
        augment=True,
        max_samples=None,
        use_torchvision=True,
    ):
        self.target_size = target_size
        self.split = split

        if use_torchvision:
            self._init_torchvision(root, split)
        else:
            self._init_manual(root, split)

        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.images):
            indices = np.random.RandomState(42).choice(
                len(self.images), max_samples, replace=False
            )
            indices.sort()
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # Count glasses
        n_glasses = sum(self.labels)
        n_no_glasses = len(self.labels) - n_glasses
        print(f"[CelebA {split}] {len(self.labels)} samples "
              f"| glasses=1: {n_glasses} | glasses=0: {n_no_glasses}")

        # Augmentation pipeline (same as GlassesDataset)
        if augment and split not in ('test', 'valid'):
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

    def _init_torchvision(self, root, split):
        """Load via torchvision.datasets.CelebA."""
        from torchvision.datasets import CelebA

        tv_split = 'train' if split == 'all' else split
        self.tv_dataset = CelebA(
            root=root,
            split=tv_split,
            target_type='attr',
            download=False,
        )

        if split == 'all':
            # Merge train + valid + test
            from torch.utils.data import ConcatDataset
            ds_train = CelebA(root=root, split='train', target_type='attr', download=False)
            ds_valid = CelebA(root=root, split='valid', target_type='attr', download=False)
            ds_test = CelebA(root=root, split='test', target_type='attr', download=False)

            self.images = []
            self.labels = []
            for ds in [ds_train, ds_valid, ds_test]:
                for i in range(len(ds)):
                    img_path = os.path.join(
                        ds.root, ds.base_folder, "img_align_celeba",
                        ds.filename[i]
                    )
                    # Eyeglasses attribute is index 15
                    label = int(ds.attr[i, 15].item())
                    # CelebA uses -1/1, convert to 0/1
                    label = 1 if label == 1 else 0
                    self.images.append(img_path)
                    self.labels.append(label)
        else:
            self.images = []
            self.labels = []
            for i in range(len(self.tv_dataset)):
                img_path = os.path.join(
                    self.tv_dataset.root, self.tv_dataset.base_folder,
                    "img_align_celeba", self.tv_dataset.filename[i]
                )
                label = int(self.tv_dataset.attr[i, 15].item())
                label = 1 if label == 1 else 0
                self.images.append(img_path)
                self.labels.append(label)

    def _init_manual(self, root, split):
        """Load from raw image folder (manual download)."""
        img_dir = os.path.join(root, 'img_align_celeba', 'img_align_celeba')
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(root, 'img_align_celeba')

        attr_file = os.path.join(root, 'list_attr_celeba.csv')

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"CelebA images not found at {img_dir}. "
                f"Download from Kaggle or Google Drive and extract."
            )

        # Parse attributes file
        glasses_map = {}
        if os.path.exists(attr_file):
            with open(attr_file, 'r') as f:
                header = f.readline().strip().split(',')
                if 'Eyeglasses' in header:
                    glasses_idx = header.index('Eyeglasses')
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > glasses_idx:
                            fname = parts[0]
                            glasses_map[fname] = 1 if int(parts[glasses_idx]) == 1 else 0
        else:
            print(f"WARNING: {attr_file} not found, all labels set to 0")

        # List all images
        all_images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        # Partition split (CelebA standard: train=162770, valid=19867, test=19962)
        if split == 'all':
            selected = all_images
        elif split == 'train':
            selected = all_images[:162770]
        elif split == 'valid':
            selected = all_images[162770:162770+19867]
        elif split == 'test':
            selected = all_images[162770+19867:]
        else:
            raise ValueError(f"split must be 'all', 'train', 'valid', or 'test'")

        self.images = [os.path.join(img_dir, f) for f in selected]
        self.labels = [glasses_map.get(f, 0) for f in selected]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"cv2 could not read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        augmented = self.transform(image=img)
        img_tensor = augmented['image'].float()

        return img_tensor, label


# ─────────────────────────────────────────────
#  Download helper
# ─────────────────────────────────────────────

def download_celeba(root='data'):
    """Download CelebA via torchvision (requires gdown)."""
    from torchvision.datasets import CelebA

    print("Downloading CelebA via torchvision...")
    print("If this fails, download manually from:")
    print("  https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print()

    dataset = CelebA(root=root, split='train', download=True)
    print(f"\nDone! {len(dataset)} training images downloaded to {root}/celeba/")


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                        help='Download CelebA via torchvision')
    parser.add_argument('--root', default='data',
                        help='Root directory for CelebA data')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual folder loading instead of torchvision')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size')
    args = parser.parse_args()

    if args.download:
        download_celeba(args.root)
    else:
        ds = CelebAFaceDataset(
            root=args.root,
            split='train',
            target_size=(64, 64),
            augment=False,
            max_samples=args.max_samples,
            use_torchvision=not args.manual,
        )

        print(f"\nDataset size: {len(ds)}")
        img, label = ds[0]
        print(f"Image shape: {img.shape}")
        print(f"Pixel range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"Label: {label}")

        # Visualise samples
        fig, axes = plt.subplots(1, 8, figsize=(20, 3))
        for i, ax in enumerate(axes):
            img, label = ds[i * (len(ds) // 8)]
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            ax.imshow(img_np)
            ax.set_title("G" if label == 1 else "NG", fontsize=9)
            ax.axis('off')
        plt.suptitle(f"CelebA samples (n={len(ds)})", fontsize=11)
        plt.tight_layout()
        plt.savefig('celeba_samples.png', dpi=150)
        print("Saved celeba_samples.png")
