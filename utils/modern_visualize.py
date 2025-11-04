"""
Modern visualization utilities following CVPR 2023-2025 trends.
Provides clean, publication-quality figure generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path


# ========== Modern colormaps ==========
def get_modern_colormap(style='viridis'):
    """
    Returns perceptually uniform colormaps popular in recent CVPR papers.

    Args:
        style: 'viridis', 'magma', 'inferno', 'plasma', 'cividis'
    """
    return plt.get_cmap(style)


def get_segmentation_palette(dataset='ade20k'):
    """
    Standard color palettes for semantic segmentation.

    Returns:
        palette: (N, 3) uint8 array
    """
    if dataset == 'ade20k':
        # ADE20K standard colors
        palette = np.array([
            [120, 120, 120],  # 0: void/background
            [180, 120, 120],  # 1: wall
            [6, 230, 230],    # 2: building
            [80, 50, 50],     # 3: sky
            [4, 200, 3],      # 4: floor
            [120, 120, 80],   # 5: tree
            [140, 140, 140],  # 6: ceiling
            [204, 5, 255],    # 7: road
            [230, 230, 230],  # 8: bed
            [4, 250, 7],      # 9: windowpane
            [224, 5, 255],    # 10: grass
        ], dtype=np.uint8)
    elif dataset == 'sunrgbd':
        # SUNRGBD / NYU40 palette
        palette = np.array([
            [0, 0, 0],        # void
            [119, 119, 119],  # wall
            [244, 243, 131],  # floor
            [137, 28, 157],   # cabinet
            [150, 255, 255],  # bed
            [54, 114, 113],   # chair
            [0, 0, 176],      # sofa
            [255, 69, 0],     # table
            [87, 112, 255],   # door
            [0, 163, 33],     # window
            [255, 150, 255],  # bookshelf
            [255, 180, 10],   # picture
            [101, 70, 86],    # counter
            [38, 230, 0],     # blinds
            [255, 120, 70],   # desk
            [117, 41, 121],   # shelves
            [150, 255, 0],    # curtain
            [132, 0, 255],    # dresser
            [24, 209, 255],   # pillow
            [191, 130, 35],   # mirror
            [219, 200, 109],  # floor mat
            [154, 62, 86],    # clothes
            [255, 190, 190],  # ceiling
            [255, 0, 255],    # books
            [152, 163, 55],   # refrigerator
            [192, 79, 212],   # television
            [230, 230, 230],  # paper
            [53, 130, 64],    # towel
            [155, 249, 152],  # shower curtain
            [87, 64, 34],     # box
            [214, 209, 175],  # whiteboard
            [170, 0, 59],     # person
            [255, 0, 0],      # night stand
            [193, 195, 234],  # toilet
            [70, 72, 115],    # sink
            [255, 255, 0],    # lamp
            [52, 57, 141],    # bathtub
            [12, 7, 134],     # bag
            [255, 228, 0],    # otherstructure
            [243, 166, 131],  # otherfurniture
            [255, 130, 0],    # otherprop
        ], dtype=np.uint8)
    else:  # default/cityscapes
        palette = np.array([
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ], dtype=np.uint8)

    return palette


# ========== Modern attention visualization ==========
def visualize_attention_modern(attn_map, rgb_img, alpha=0.6, cmap='viridis',
                               smooth=True, normalize=True):
    """
    Modern attention visualization with perceptually uniform colormap.

    Args:
        attn_map: (H, W) attention map [0, 1]
        rgb_img: (H, W, 3) RGB image uint8
        alpha: blending weight for overlay
        cmap: 'viridis', 'magma', 'inferno', 'plasma'
        smooth: apply Gaussian smoothing
        normalize: normalize to [0, 1]

    Returns:
        overlay: (H, W, 3) RGB image uint8
    """
    H, W = attn_map.shape

    # Resize if needed
    if rgb_img.shape[:2] != (H, W):
        rgb_img = cv2.resize(rgb_img, (W, H))

    # Smooth
    if smooth:
        from scipy.ndimage import gaussian_filter
        attn_map = gaussian_filter(attn_map, sigma=1.0)

    # Normalize
    if normalize:
        amin, amax = attn_map.min(), attn_map.max()
        if amax > amin:
            attn_map = (attn_map - amin) / (amax - amin)

    # Apply modern colormap
    cmap_func = get_modern_colormap(cmap)
    heat_color = (cmap_func(attn_map)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (alpha * heat_color + (1 - alpha) * rgb_img).astype(np.uint8)

    return overlay


# ========== Multi-panel figure generation ==========
def create_segmentation_grid(rgb, pred, gt=None, attention=None,
                             class_names=None, palette='sunrgbd',
                             title=None, save_path=None, dpi=150):
    """
    Create modern multi-panel visualization grid.

    Args:
        rgb: (H, W, 3) RGB image uint8
        pred: (H, W) predicted labels
        gt: (H, W) ground truth labels (optional)
        attention: (H, W) attention map (optional)
        class_names: list of class names
        palette: color palette name or (N, 3) array
        title: figure title
        save_path: output path
        dpi: figure DPI

    Returns:
        fig: matplotlib figure
    """
    # Setup
    if isinstance(palette, str):
        colors = get_segmentation_palette(palette)
    else:
        colors = palette

    # Determine grid size
    n_panels = 2  # rgb + pred
    if gt is not None:
        n_panels += 1
    if attention is not None:
        n_panels += 1

    # Create figure with modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    # Remove ticks and spines for cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    panel_idx = 0

    # Panel 1: RGB
    axes[panel_idx].imshow(rgb)
    axes[panel_idx].set_title('Input Image', fontsize=14, fontweight='bold')
    panel_idx += 1

    # Panel 2: Prediction
    pred_colored = colors[pred]
    axes[panel_idx].imshow(pred_colored)
    axes[panel_idx].set_title('Prediction', fontsize=14, fontweight='bold')
    panel_idx += 1

    # Panel 3: Ground Truth (if available)
    if gt is not None:
        gt_colored = colors[gt]
        axes[panel_idx].imshow(gt_colored)
        axes[panel_idx].set_title('Ground Truth', fontsize=14, fontweight='bold')
        panel_idx += 1

    # Panel 4: Attention (if available)
    if attention is not None:
        attn_overlay = visualize_attention_modern(attention, rgb, cmap='magma')
        axes[panel_idx].imshow(attn_overlay)
        axes[panel_idx].set_title('Attention Map', fontsize=14, fontweight='bold')

    # Add legend if class names provided
    if class_names is not None:
        unique_labels = np.unique(pred)
        patches = []
        for label in unique_labels:
            if label < len(class_names) and label < len(colors):
                color = colors[label] / 255.0
                patches.append(mpatches.Patch(color=color, label=class_names[label]))

        # Place legend outside plot area
        if len(patches) <= 10:
            fig.legend(handles=patches, loc='lower center',
                      ncol=min(len(patches), 5),
                      bbox_to_anchor=(0.5, -0.05),
                      frameon=True, fontsize=10)

    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    return fig


# ========== Comparison figure for multiple methods ==========
def create_method_comparison(rgb, predictions_dict, gt=None,
                             class_names=None, palette='sunrgbd',
                             save_path=None, dpi=150):
    """
    Create comparison figure for multiple methods (CVPR-style).

    Args:
        rgb: (H, W, 3) RGB image
        predictions_dict: {method_name: pred_map} dictionary
        gt: (H, W) ground truth
        class_names: list of class names
        palette: color palette
        save_path: output path

    Returns:
        fig: matplotlib figure
    """
    if isinstance(palette, str):
        colors = get_segmentation_palette(palette)
    else:
        colors = palette

    n_methods = len(predictions_dict)
    n_panels = 1 + n_methods + (1 if gt is not None else 0)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Input
    axes[0].imshow(rgb)
    axes[0].set_title('Input', fontsize=12, fontweight='bold')

    # Methods
    for i, (method_name, pred) in enumerate(predictions_dict.items(), 1):
        pred_colored = colors[pred]
        axes[i].imshow(pred_colored)
        axes[i].set_title(method_name, fontsize=12, fontweight='bold')

    # GT
    if gt is not None:
        gt_colored = colors[gt]
        axes[-1].imshow(gt_colored)
        axes[-1].set_title('Ground Truth', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    return fig


# ========== Usage Example ==========
if __name__ == '__main__':
    # Example: Create a modern visualization
    H, W = 480, 640
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    pred = np.random.randint(0, 40, (H, W), dtype=np.uint8)
    gt = np.random.randint(0, 40, (H, W), dtype=np.uint8)
    attention = np.random.rand(H, W).astype(np.float32)

    class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair']

    # Single image visualization
    fig = create_segmentation_grid(
        rgb, pred, gt, attention,
        class_names=class_names,
        palette='sunrgbd',
        title='SUNRGBD Semantic Segmentation',
        save_path='modern_viz_example.png'
    )

    print("✓ Example visualization saved to modern_viz_example.png")
