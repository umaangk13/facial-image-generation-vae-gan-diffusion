# Facial Image Generation: VAE, GAN, and Diffusion

This repository contains the implementation of three distinct generative models—**Variational Autoencoder (VAE)**, **Generative Adversarial Network (GAN)**, and **Denoising Diffusion Probabilistic Model (DDPM)**—designed to generate 64x64 face images. 

The primary objective is to implement these architectures from scratch using PyTorch, perform class-conditional generation (faces with and without glasses), and systematically evaluate their performance using the Fréchet Inception Distance (FID) and automated ablation studies.

Developed for **CS F437: Generative AI** (Assignment 1).

---

## 🚀 Key Features

*   **Three Core Architectures:**
    *   **Conditional VAE (`vae.py`)**: Uses label embeddings injected into both the encoder and decoder to learn class-specific latent spaces.
    *   **DCGAN (`gan.py` & `gan2.py`)**: Implementations of both Unconditional and Conditional DCGAN architectures.
    *   **DDPM (`diffusion.py`)**: A conditional Denoising Diffusion Probabilistic Model utilizing a custom UNet with sinusoidal timestep embeddings.
*   **Modern Architectural Tweaks:** Custom **Residual Blocks** (`x = x + layer(x)`) implemented across the VAE Encoder/Decoder and the GAN Generator/Discriminator to solve vanishing gradients, smooth the loss landscape, and accelerate convergence.
*   **Automated Ablation Studies:** Dedicated scripts (`ablation_*.py`) that automatically run grid searches over ≥5 hyperparameters per model, save progression images, and output markdown summary tables. Resume-capability built in.
*   **Automated Evaluation:** Evaluation pipeline utilizing `torch-fidelity` to compute the Fréchet Inception Distance (FID) score (`evaluate.py`, `eval_fid_from_checkpoints.py`) to benchmark the quality and diversity of generated faces against the real dataset.
*   **Visualizations & Demos:** Includes utilities to automatically generate model output demos (`generate_demo.py`) and save architectural block diagrams.

## 📁 Repository Structure

```text
├── dataset.py                # Albumentations data pipeline & GlassesDataset definition
├── vae.py                    # CVAE architecture (Encoder, Decoder, Loss)
├── gan.py                    # Unconditional DCGAN (Generator, Discriminator, ResBlocks)
├── gan2.py                   # Conditional cGAN architecture
├── diffusion.py              # DDPM noise schedule, training loop, and UNet
├── evaluate.py               # FID score calculation and model benchmarking
├── ablation_vae.py           # Automated ablation runner for VAE
├── ablation_gan.py           # Automated ablation runner for GAN
├── ablation_diffusion.py     # Automated ablation runner for Diffusion
├── eval_fid_from_checkpoints.py # Batch offline FID calculator for ablation checkpoints
├── generate_demo.py          # Unified image generation demo script for all models
├── describe_models.py        # CLI tool to print rich layer-by-layer parameter summaries
├── visuals/                  # Organized output folders (`diffusion_results/`, `gan_results/`, etc.)
├── final_train.csv           # Dataset labels (binary: has_glasses)
├── requirements.txt          # Python package dependencies
└── README.md                 # You are here
```

> **Note:** Checkpoint files (`.pth`) and the raw `images/` directory are excluded from version control due to GitHub storage limits.

---

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/umaangk13/facial-image-generation-vae-gan-diffusion.git
   cd facial-image-generation-vae-gan-diffusion
   ```

2. Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure the `images/` directory (containing the 64x64 face dataset) and `final_train.csv` are placed in the root of the project.

---

## 🏃‍♂️ Running the Code

### 1. Training a Single Model
You can run any of the core model files directly to start a standard training run (50 epochs) using the baseline configuration. This will output progression images every 10 epochs and save the `_best.pth` and `_final.pth` checkpoints.

```bash
python vae.py
python gan.py
python diffusion.py
```

### 2. Running Automated Ablation Studies
To run the automated experiments, use the ablation scripts. These will systematically alter hyperparameters (like learning rate, latent dim, ResBlock count), train the model, compute the SSIM, and save the results in an `ablation_[model]/` dictionary. 

*If interrupted, simply re-run the script—it will load existing `result.json` files and skip completed experiments.*

```bash
python ablation_vae.py
python ablation_gan.py
python ablation_diffusion.py
```

### 3. Running the Generation Demo (Inference)
To run a unified inference script that loads the best trained weights for the VAE, GAN, and Diffusion models and generates a grid of sample outputs automatically:
```bash
python generate_demo.py
```
*Generated images are automatically saved to the `demo_generated/` folder.*

### 4. Evaluating FID Metrics
To recalculate FID scores directly from existing ablation checkpoints without retraining:
```bash
python eval_fid_from_checkpoints.py
```

To run the automated FID benchmark against a specific model checkpoint manually:
```bash
python evaluate.py --model vae --checkpoint vae_best.pth
```
*(Make sure to adjust the `--model` flag to `vae`, `gan`, or `diffusion`)*

---
*Created for BITS Pilani CS F437 - Generative AI.*
