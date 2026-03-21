"""
Visualise model architectures as block diagrams (left-to-right flow).
Generates one PNG per model showing the layer flow.

Usage:
    python visualise_architectures.py
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────
#  Drawing Helpers
# ─────────────────────────────────────────────

# Color palette
COLORS = {
    'input':     '#4CAF50',   # green
    'conv':      '#2196F3',   # blue
    'resblock':  '#1565C0',   # dark blue
    'bn':        '#90CAF9',   # light blue
    'activation':'#FF9800',   # orange
    'pool':      '#9C27B0',   # purple
    'fc':        '#E91E63',   # pink
    'embed':     '#00BCD4',   # cyan
    'output':    '#F44336',   # red
    'upsample':  '#8BC34A',   # light green
    'noise':     '#607D8B',   # grey
    'special':   '#FF5722',   # deep orange
    'concat':    '#795548',   # brown
    'arrow':     '#424242',   # dark grey
}


def draw_block_diagram(ax, blocks, title, y_center=0.5):
    """
    Draw a left-to-right block diagram.
    blocks: list of (label, sublabel, color_key, height_ratio)
    """
    ax.set_xlim(-0.5, len(blocks) + 0.5)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    block_width = 0.7
    gap = 0.3
    total_width = len(blocks) * (block_width + gap) - gap

    for i, (label, sublabel, color_key, h_ratio) in enumerate(blocks):
        x = i
        h = 0.5 * h_ratio
        y = y_center - h / 2

        color = COLORS.get(color_key, '#999999')

        # Draw block
        rect = FancyBboxPatch(
            (x - block_width / 2, y), block_width, h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white',
            linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)

        # Main label
        ax.text(x, y_center + 0.02, label,
                ha='center', va='center', fontsize=7,
                fontweight='bold', color='white',
                wrap=True)

        # Sub label (dimensions etc)
        if sublabel:
            ax.text(x, y - 0.08, sublabel,
                    ha='center', va='top', fontsize=5,
                    color='white', alpha=0.85)

        # Arrow to next block
        if i < len(blocks) - 1:
            ax.annotate('',
                        xy=(x + block_width / 2 + gap * 0.7, y_center),
                        xytext=(x + block_width / 2 + 0.02, y_center),
                        arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                        lw=1.5))


# ─────────────────────────────────────────────
#  Model Definitions (as block lists)
# ─────────────────────────────────────────────

def get_vae_encoder_blocks():
    return [
        ('Input',       '3x64x64',    'input',     1.0),
        ('Label\nEmbed', 'class->32d', 'embed',     0.7),
        ('Concat',      '4x64x64',    'concat',    0.8),
        ('ResBlock\nConv1',  '4->32\n32x32', 'resblock', 1.0),
        ('ResBlock\nConv2',  '32->64\n16x16', 'resblock', 1.0),
        ('ResBlock\nConv3',  '64->128\n8x8',  'resblock', 1.0),
        ('ResBlock\nConv4',  '128->256\n4x4', 'resblock', 1.0),
        ('Flatten',     '256*4*4',    'special',   0.6),
        ('FC: mu',      '->128',      'fc',        0.8),
        ('FC: logvar',  '->128',      'fc',        0.8),
    ]

def get_vae_decoder_blocks():
    return [
        ('z ~ N(0,I)',  '128d',       'noise',     0.7),
        ('Label\nEmbed', 'class->32d', 'embed',     0.7),
        ('Concat',      '160d',       'concat',    0.6),
        ('FC',          '->256*4*4',  'fc',        0.8),
        ('Reshape',     '256x4x4',    'special',   0.6),
        ('ResBlock\nDeconv1', '256->128\n8x8',  'resblock', 1.0),
        ('ResBlock\nDeconv2', '128->64\n16x16', 'resblock', 1.0),
        ('ResBlock\nDeconv3', '64->32\n32x32',  'resblock', 1.0),
        ('Deconv4',     '32->3\n64x64', 'conv',    0.9),
        ('Tanh',        '[-1,1]',     'activation', 0.6),
        ('Output',      '3x64x64',   'output',    0.8),
    ]

def get_gan_generator_blocks():
    return [
        ('z ~ N(0,I)',  '128d',       'noise',     0.8),
        ('FC+BN',       '->512*4*4',  'fc',        0.9),
        ('Reshape',     '512x4x4',    'special',   0.6),
        ('ResBlock\nDeconv1', '512->256\n8x8',  'resblock', 1.0),
        ('ResBlock\nDeconv2', '256->128\n16x16', 'resblock', 1.0),
        ('ResBlock\nDeconv3', '128->64\n32x32',  'resblock', 1.0),
        ('Deconv4',     '64->3\n64x64', 'conv',    0.9),
        ('Tanh',        '[-1,1]',     'activation', 0.6),
        ('Output',      '3x64x64',   'output',    0.8),
    ]

def get_gan_discriminator_blocks():
    return [
        ('Input',       '3x64x64',   'input',     0.8),
        ('ResBlock1',   '3->64\n32x32',  'resblock', 1.0),
        ('ResBlock2',   '64->128\n16x16', 'resblock', 1.0),
        ('ResBlock3',   '128->256\n8x8',  'resblock', 1.0),
        ('ResBlock4',   '256->512\n4x4',  'resblock', 1.0),
        ('Flatten',     '512*4*4',   'special',   0.6),
        ('FC',          '->1',       'fc',        0.8),
        ('Sigmoid',     '[0,1]',     'activation', 0.6),
        ('Real/Fake',   'prob',      'output',    0.7),
    ]

def get_diffusion_unet_blocks():
    return [
        ('Noisy\nImage', 'x_t 3x64x64', 'input',   0.8),
        ('Init Conv',   '3->64',      'conv',      0.7),

        # Encoder
        ('ResBlock\nDown1', '64->64\n64x64', 'resblock', 1.0),
        ('Downsample',  '64x32x32',   'pool',      0.6),
        ('ResBlock\nDown2', '64->128\n32x32', 'resblock', 1.0),
        ('Downsample',  '128x16x16',  'pool',      0.6),
        ('ResBlock\nDown3', '128->256\n16x16','resblock', 1.0),
        ('Downsample',  '256x8x8',    'pool',      0.6),
        ('ResBlock\nDown4', '256->512\n8x8', 'resblock', 1.0),

        # Bottleneck
        ('Bottleneck',  '512x8x8',    'special',   0.9),

        # Decoder
        ('ResBlock\nUp4', '512->512\n8x8', 'resblock', 1.0),
        ('Upsample',   '512x16x16',  'upsample',  0.6),
        ('ResBlock\nUp3', '512->256\n16x16','resblock', 1.0),
        ('Upsample',   '256x32x32',  'upsample',  0.6),
        ('ResBlock\nUp2', '256->128\n32x32','resblock', 1.0),
        ('Upsample',   '128x64x64',  'upsample',  0.6),
        ('ResBlock\nUp1', '128->64\n64x64', 'resblock', 1.0),

        ('Final Conv',  '64->3',     'conv',      0.7),
        ('Pred Noise',  'eps 3x64x64','output',   0.8),
    ]


def add_conditioning_annotation(ax, blocks, t_idx, label_idx, title_text):
    """Add time + class conditioning annotation above the diagram."""
    mid = len(blocks) / 2
    ax.text(mid, 1.15,
            '+ Time Embedding (sinusoidal)  +  Class Label Embedding',
            ha='center', va='center', fontsize=8,
            style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8EAF6',
                      edgecolor='#7986CB', alpha=0.8))


# ─────────────────────────────────────────────
#  Main: Generate all diagrams
# ─────────────────────────────────────────────

if __name__ == '__main__':

    output_dir = '.'

    # ── 1. CVAE ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))

    enc_blocks = get_vae_encoder_blocks()
    dec_blocks = get_vae_decoder_blocks()

    draw_block_diagram(ax1, enc_blocks, 'CVAE Encoder: Image + Label -> mu, logvar')
    draw_block_diagram(ax2, dec_blocks, 'CVAE Decoder: z + Label -> Generated Image')

    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_vae.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

    # ── 2. GAN ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))

    gen_blocks = get_gan_generator_blocks()
    disc_blocks = get_gan_discriminator_blocks()

    draw_block_diagram(ax1, gen_blocks, 'GAN Generator: Noise z -> Fake Image')
    draw_block_diagram(ax2, disc_blocks, 'GAN Discriminator: Image -> Real/Fake')

    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_gan.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

    # ── 3. Diffusion UNet ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(28, 4))

    unet_blocks = get_diffusion_unet_blocks()
    draw_block_diagram(ax, unet_blocks, 'DDPM Conditional UNet: Noisy Image x_t -> Predicted Noise eps')
    add_conditioning_annotation(ax, unet_blocks, 0, 0, '')

    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture_diffusion.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

    # ── Legend ────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    ax.axis('off')
    ax.set_title('Legend', fontsize=12, fontweight='bold')

    legend_items = [
        ('Input/Output', 'input'),
        ('ResBlock', 'resblock'),
        ('Conv/Deconv', 'conv'),
        ('FC Layer', 'fc'),
        ('Activation', 'activation'),
        ('Down/Pooling', 'pool'),
        ('Upsample', 'upsample'),
        ('Embedding', 'embed'),
        ('Concat/Reshape', 'special'),
        ('Noise', 'noise'),
    ]

    for i, (label, ckey) in enumerate(legend_items):
        x = i * 1.2
        rect = FancyBboxPatch(
            (x, 0.2), 0.8, 0.5,
            boxstyle="round,pad=0.05",
            facecolor=COLORS[ckey], edgecolor='white',
            linewidth=1, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x + 0.4, 0.45, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    ax.set_xlim(-0.5, len(legend_items) * 1.2)
    ax.set_ylim(-0.1, 1.0)

    path = os.path.join(output_dir, 'architecture_legend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

    print("\nAll architecture diagrams saved!")
