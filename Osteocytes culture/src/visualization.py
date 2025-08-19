# visualization.py

import matplotlib.pyplot as plt
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

def plot_histograms(image: np.ndarray, areas: list, dendritic_lengths: list, eccentricities: list, solidities: list, output_path: str, dpi: int = 150):
    """Plot histograms for intensity and cell metrics using seaborn.
    Args:
        image (np.ndarray): Input image for intensity histogram.
        areas (list): List of cell areas.
        dendritic_lengths (list): List of dendritic lengths.
        eccentricities (list): List of eccentricities.
        solidities (list): List of solidities.
        output_path (str): Path to save PNG.
        dpi (int): Resolution (default: 150).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D NumPy array.")
    for name, lst in [('areas', areas), ('dendritic_lengths', dendritic_lengths),
                      ('eccentricities', eccentricities), ('solidities', solidities)]:
        if not lst:
            logger.warning(f"Empty {name} list for {output_path}")
            raise ValueError(f"{name} list is empty")
    logger.debug(f"Plotting histograms: {len(areas)} cells, image shape: {image.shape}")
    sns.set_style('whitegrid')
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    sns.histplot(image.ravel(), bins=128, kde=True)
    plt.axvline(np.mean(image.ravel()), color='red', linestyle='--', label='Mean')
    plt.title('Image Intensity')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.legend()
    plt.subplot(1, 4, 2)
    sns.histplot(areas, bins=min(20, len(areas)//2 or 1), kde=True)
    plt.axvline(np.mean(areas), color='red', linestyle='--', label='Mean')
    plt.title('Cell Area')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.subplot(1, 4, 3)
    sns.histplot(dendritic_lengths, bins=min(20, len(dendritic_lengths)//2 or 1), kde=True)
    plt.axvline(np.mean(dendritic_lengths), color='red', linestyle='--', label='Mean')
    plt.title('Dendritic Length')
    plt.xlabel('Length (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.subplot(1, 4, 4)
    sns.histplot(eccentricities, bins=min(20, len(eccentricities)//2 or 1), kde=True)
    plt.axvline(np.mean(eccentricities), color='red', linestyle='--', label='Mean')
    plt.title('Eccentricity')
    plt.xlabel('Eccentricity')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved histograms to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save histograms to {output_path}: {e}")
    plt.close()

def plot_skeleton_overlays(labeled: np.ndarray, cell_metrics: pd.DataFrame, output_dir: str, percentile: float):
    """Plot skeleton overlays for each cell, showing cell mask and skeleton.
    Args:
        labeled (np.ndarray): Labeled segmentation mask.
        cell_metrics (pd.DataFrame): Cell metrics DataFrame with index for labels.
        output_dir (str): Directory to save skeleton overlay plots.
        percentile (float): Percentile used for segmentation (for logging).
    """
    if not isinstance(labeled, np.ndarray) or labeled.ndim != 2 or labeled.size == 0:
        logger.error(f"Invalid labeled mask shape: {labeled.shape if isinstance(labeled, np.ndarray) else type(labeled)}")
        raise ValueError("Labeled mask must be a non-empty 2D NumPy array.")
    try:
        logger.info("Analyzing dendrites for skeleton overlays")
        # Relabel sequentially to ensure labels start from 1
        labeled, _, _ = skimage.segmentation.relabel_sequential(labeled)
        logger.debug(f"Unique labels: {np.unique(labeled)}")
        logger.debug(f"Label counts: {np.bincount(labeled.ravel())}")
        logger.debug(f"Cell metrics index: {cell_metrics.index.tolist()}")
        
        # Find bounding boxes for each cell
        cell_slices = ndimage.find_objects(labeled)
        logger.info(f"Number of cells: {len(cell_slices)}, percentile: {percentile}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(cell_slices):
            if s is None:
                logger.warning(f"No slice found for label {i+1}. Skipping.")
                continue
            view = labeled[s]
            cell_mask = (view == i + 1)
            if np.any(np.array(cell_mask.shape) > 100):
                logger.debug(f"Skipping label {i+1} due to slice dimensions {cell_mask.shape} > 100")
                continue
            try:
                # Skeletonize the cell mask
                skeleton = skimage.morphology.skeletonize(cell_mask)
                # Plot overlay (only cell mask and skeleton, no combined image)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.matshow(cell_mask, cmap="Oranges", alpha=0.5)
                ax.matshow(skeleton, cmap="bone_r", alpha=0.1)
                ax.axis('off')
                output_path = Path(output_dir) / f"skeleton_overlay_label_{i+1}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved skeleton overlay for label {i+1} to {output_path}")
            except Exception as e:
                logger.error(f"Error plotting skeleton overlay for label {i+1}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in plot_skeleton_overlays: {e}")