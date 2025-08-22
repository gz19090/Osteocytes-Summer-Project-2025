# visualization.py

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
import seaborn as sns
import logging
from skimage.measure import regionprops
from scipy import ndimage
import skimage.segmentation
import skimage.morphology
from skimage.morphology import skeletonize, opening, disk, remove_small_objects
from skimage.segmentation import relabel_sequential, find_boundaries
from skimage.measure import find_contours

# Set up logging
logger = logging.getLogger(__name__)

def plot_edge_filters(image: np.ndarray, output_path: str, dpi: int = 150):
    """Plot results of edge filters for debugging cell boundary detection.
    Args:
        image (np.ndarray): Input image (2D grayscale array).
        output_path (str): Path to save PNG.
        dpi (int): Resolution (default: 150).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")
    logger.debug(f"Plotting edge filters for image shape: {image.shape}")
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, f in enumerate(edge_filters):
        filtered = f(image)
        min_val, max_val = filtered.min(), filtered.max()
        if max_val > min_val:
            filtered = (filtered - min_val) / (max_val - min_val)
        axes[idx].imshow(filtered, cmap='gray')
        axes[idx].set_title(f.__name__)
        axes[idx].axis('off')
        logger.debug(f"Filter {f.__name__} range: {min_val:.2f}, {max_val:.2f}")
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved edge filters to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save edge filters to {output_path}: {e}")
    plt.close()

def plot_combined_image(image: np.ndarray, combined: np.ndarray, weights: list, output_path: str, dpi: int = 150):
    """Plot combined edge-filtered image with weights.
    Args:
        image (np.ndarray): Original image (unused, for API consistency).
        combined (np.ndarray): Combined edge-filtered image.
        weights (list): Weights for edge filters (length 6).
        output_path (str): Path to save PNG.
        dpi (int): Resolution (default: 150).
    """
    if not isinstance(combined, np.ndarray) or combined.ndim != 2 or combined.size == 0:
        raise ValueError("Combined image must be a non-empty 2D NumPy array.")
    if len(weights) != 6:
        raise ValueError(f"Expected 6 weights, got {len(weights)}")
    logger.debug(f"Plotting combined image shape: {combined.shape}, weights: {np.round(weights, 3)}")
    plt.figure(figsize=(20, 10))
    plt.imshow(combined, cmap='gray')
    filter_names = ['Sobel', 'Scharr', 'Prewitt', 'Roberts', 'Farid', 'Laplace']
    title = f'Best Linear Combination\nWeights: {", ".join(f"{n}={w:.3f}" for n, w in zip(filter_names, weights))}'
    plt.title(title)
    plt.axis('off')
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved combined image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save combined image to {output_path}: {e}")
    plt.close()

def plot_contours(image: np.ndarray, contours: list, output_path: str, dpi: int = 150):
    """Plot contours on the image for cell boundary visualization.
    Args:
        image (np.ndarray): Input image (typically edge-filtered).
        contours (list): List of contours (Nx2 arrays from skimage.measure.find_contours).
        output_path (str): Path to save PNG.
        dpi (int): Resolution (default: 150).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D NumPy array.")
    if not contours:
        logger.warning(f"No contours provided for {output_path}")
        raise ValueError("Contours list is empty")
    logger.debug(f"Plotting {len(contours)} contours on image shape: {image.shape}")
    plt.figure(figsize=(20, 10))
    plt.imshow(image, cmap='gray')
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    plt.plot([10, 60], [10, 10], color='white', linewidth=4)
    plt.text(10, 20, '25 µm', color='white', fontsize=12)
    plt.axis('off')
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved contours to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save contours to {output_path}: {e}")
    plt.close()

def plot_segmentation(image: np.ndarray, combined: np.ndarray, labeled: np.ndarray, output_path: str, dpi: int = 150, fontsize: int = 6):
    """Plot segmentation results with optional cell ID annotations.
    Args:
        image (np.ndarray): Preprocessed image.
        combined (np.ndarray): Edge-filtered image.
        labeled (np.ndarray): Labeled segmentation mask.
        output_path (str): Base path for PNGs.
        dpi (int): Resolution (default: 150).
        fontsize (int): Font size for cell ID annotations (default: 6).
    """
    for name, arr in [('image', image), ('combined', combined), ('labeled', labeled)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 2D NumPy array.")
    num_cells = len(np.unique(labeled)) - 1
    if num_cells == 0:
        logger.warning(f"No cells detected for {output_path}")
        raise ValueError("No cells in labeled mask")
    logger.debug(f"Plotting segmentation: {num_cells} cells, image shape: {image.shape}")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(combined, cmap='gray')
    plt.title('Edge-Filtered Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(labeled, cmap='nipy_spectral')
    plt.title(f'Labeled Cells ({num_cells})')
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved segmentation (without IDs) to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save segmentation to {output_path}: {e}")
    plt.close()
    output_path_labeled = str(Path(output_path).with_stem(Path(output_path).stem + '_labeled'))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(combined, cmap='gray')
    plt.title('Edge-Filtered Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(labeled, cmap='nipy_spectral')
    regions = regionprops(labeled)
    for region in regions:
        y, x = region.centroid
        plt.text(x, y, str(region.label), color='white', fontsize=fontsize, ha='center', va='center')
    plt.title(f'Labeled Cells ({num_cells})')
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_path_labeled, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved segmentation (with IDs) to {output_path_labeled}")
    except Exception as e:
        logger.error(f"Failed to save labeled segmentation to {output_path_labeled}: {e}")
    plt.close()

def plot_histograms(image: np.ndarray, areas: list, dendrite_counts: list, eccentricities: list, solidities: list, output_path: str, dpi: int = 150):
    """Plot histograms for intensity and cell metrics (using dendrite COUNT)."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D NumPy array.")
    for name, lst in [('areas', areas), ('dendrite_counts', dendrite_counts),
                      ('eccentricities', eccentricities), ('solidities', solidities)]:
        if lst is None or len(lst) == 0:
            logger.warning(f"Empty {name} list for {output_path}")
            raise ValueError(f"{name} list is empty")

    logger.debug(f"Plotting histograms: {len(areas)} cells, image shape: {image.shape}")
    sns.set_style('whitegrid')
    plt.figure(figsize=(16, 4))

    # Intensity
    plt.subplot(1, 4, 1)
    sns.histplot(image.ravel(), bins=128, kde=True)
    plt.axvline(np.mean(image.ravel()), color='red', linestyle='--', label='Mean')
    plt.title('Image Intensity'); plt.xlabel('Intensity'); plt.ylabel('Count'); plt.legend()

    # Area
    plt.subplot(1, 4, 2)
    sns.histplot(areas, bins=min(20, len(areas)//2 or 1), kde=True)
    plt.axvline(np.mean(areas), color='red', linestyle='--', label='Mean')
    plt.title('Cell Area'); plt.xlabel('Area (pixels)'); plt.ylabel('Count'); plt.legend()

    # Dendrite COUNT (discrete)
    plt.subplot(1, 4, 3)
    dmin, dmax = int(np.min(dendrite_counts)), int(np.max(dendrite_counts))
    bins = np.arange(dmin, dmax + 2) - 0.5  # integer bins
    sns.histplot(dendrite_counts, bins=bins, discrete=True)
    plt.axvline(np.mean(dendrite_counts), color='red', linestyle='--', label='Mean')
    plt.title('Dendrite Count'); plt.xlabel('# Dendrites'); plt.ylabel('Cells'); plt.legend()

    # Eccentricity
    plt.subplot(1, 4, 4)
    sns.histplot(eccentricities, bins=min(20, len(eccentricities)//2 or 1), kde=True)
    plt.axvline(np.mean(eccentricities), color='red', linestyle='--', label='Mean')
    plt.title('Eccentricity'); plt.xlabel('Eccentricity'); plt.ylabel('Count'); plt.legend()

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved histograms to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save histograms to {output_path}: {e}")
    plt.close()

def _protrusion_skeleton(cell_mask: np.ndarray,
                         soma_frac: float = 0.18,
                         min_component_size: int = 8) -> np.ndarray:
    """Skeleton of dendrite-like protrusions only (soma removed by opening)."""
    cell = cell_mask.astype(bool, copy=False)
    if cell.sum() == 0:
        return np.zeros_like(cell, dtype=bool)
    area = int(cell.sum())
    equiv_d = 2.0 * np.sqrt(area / np.pi)
    r = max(2, int(round(soma_frac * equiv_d)))  # adaptive soma radius
    core = opening(cell, disk(r))
    protrusions = cell & ~core
    if protrusions.any():
        protrusions = remove_small_objects(protrusions, min_size=min_component_size)
    return skeletonize(protrusions)

def plot_skeleton_overlays(
    labeled: np.ndarray,
    image_processed: np.ndarray,  # grayscale background (e.g., 'combined')
    cell_metrics: pd.DataFrame,
    output_dir: str,
    image_original: np.ndarray = None,
    soma_frac: float = 0.18,
    min_component_size: int = 8,
    min_len_px: int = 5,
    alpha_mask: float = 0.38,
    # colors
    fill_color: str = "#FFA726", # orange 
    outline_color: str = "saddlebrown",
):
    """
    LEFT  : FULL skeleton (light gray) on grayscale background
    RIGHT : PROTRUSION skeleton (white; counted) on grayscale background
    Cells are filled and contoured with brown.
    """
    if not isinstance(labeled, np.ndarray) or labeled.ndim != 2 or labeled.size == 0:
        raise ValueError("Labeled mask must be a non-empty 2D NumPy array.")
    if not isinstance(image_processed, np.ndarray) or image_processed.ndim != 2 or image_processed.size == 0:
        raise ValueError("image_processed must be a non-empty 2D NumPy array.")
    if 'dendrite_count' not in cell_metrics.columns:
        raise ValueError("cell_metrics must include 'dendrite_count'.")

    metrics_idx = (cell_metrics['cell_id'].values if 'cell_id' in cell_metrics.columns
                   else cell_metrics.index.values).astype(int)
    labeled, _, _ = relabel_sequential(labeled)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dendrite_map = {int(cid): int(cnt) for cid, cnt in zip(metrics_idx, cell_metrics['dendrite_count'].values)}

    # single-color cmaps (1 -> color; 0 -> transparent via masked arrays)
    fill_cmap   = ListedColormap([fill_color])
    outline_cmap = ListedColormap([outline_color])
    full_cmap   = ListedColormap(['#DCDCDC'])  # light gray
    prot_cmap   = ListedColormap(['#FFFFFF'])  # white

    cell_slices = ndimage.find_objects(labeled)
    for i, s in enumerate(cell_slices, start=1):
        if s is None:
            continue
        rsl, csl = s
        r0, r1 = rsl.start, rsl.stop
        c0, c1 = csl.start, csl.stop
        pad = 5
        r0p = max(0, r0 - pad); r1p = min(labeled.shape[0], r1 + pad)
        c0p = max(0, c0 - pad); c1p = min(labeled.shape[1], c1 + pad)

        lab_view = labeled[r0p:r1p, c0p:c1p]
        bg_view  = image_processed[r0p:r1p, c0p:c1p]
        cell_mask = (lab_view == i)

        # 1‑px skeletons
        skel_full = skeletonize(cell_mask)
        skel_prot = _protrusion_skeleton(cell_mask, soma_frac=soma_frac, min_component_size=min_component_size)
        if skel_prot.sum() < min_len_px:
            skel_prot[:] = False

        # boundaries + masked arrays
        outline = find_boundaries(cell_mask, mode='outer')
        mask_ma = np.ma.masked_where(~cell_mask, cell_mask)
        outl_ma = np.ma.masked_where(~outline, outline)
        full_ma = np.ma.masked_where(~skel_full, skel_full)
        prot_ma = np.ma.masked_where(~skel_prot, skel_prot)

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
        panels = [
            ("FULL skeleton", full_ma, full_cmap),
            (f"PROTRUSION skeleton (counted: {dendrite_map.get(i, 0)})", prot_ma, prot_cmap),
        ]

        for ax, (subtitle, skel_ma, skel_cmap) in zip(axes, panels):
            ax.imshow(bg_view, cmap="gray", interpolation="nearest")  # original grayscale
            # filled cell with chosen color
            ax.imshow(mask_ma, cmap=fill_cmap, alpha=alpha_mask, interpolation="nearest", vmin=0, vmax=1)
            # crisp colored contour
            ax.imshow(outl_ma, cmap=outline_cmap, interpolation="nearest", vmin=0, vmax=1)
            # pixel-exact skeleton overlay
            ax.imshow(skel_ma, cmap=skel_cmap, interpolation="nearest", vmin=0, vmax=1)

            ax.set_title(f"Cell {i} | {subtitle}")
            ax.axis("off")

        out_path = Path(output_dir) / f"skeleton_overlay_label_{i}.png"
        try:
            plt.savefig(out_path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)