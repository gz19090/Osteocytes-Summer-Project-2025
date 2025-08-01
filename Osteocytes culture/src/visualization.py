# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
import seaborn as sns
import logging
from skimage.measure import regionprops

# Set up logging for debugging
logger = logging.getLogger(__name__)

def plot_edge_filters(image: np.ndarray, output_path: str, dpi: int = 150):
    """
    Plot results of edge filters for debugging cell boundary detection.

    Applies six edge detection filters (Sobel, Scharr, Prewitt, Roberts, Farid, Laplace) to the input
    image, normalizes each output, and displays in a 2x3 subplot grid. Validates input shape.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
        output_path (str): Path to save PNG (e.g., results/figures/.../edge_filters.png).
        dpi (int): Resolution (default: 150 for debugging, use 300 for publication).

    Raises:
        ValueError: If image is not 2D or empty.
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

    Displays the combined edge-filtered image from linear regression, with weights in the title.
    Validates inputs; includes filter names for clarity.

    Args:
        image (np.ndarray): Original image (unused, for API consistency).
        combined (np.ndarray): Combined edge-filtered image.
        weights (list): Weights for edge filters (length 6: Sobel, Scharr, etc.).
        output_path (str): Path to save PNG (e.g., results/figures/.../combined.png).
        dpi (int): Resolution (default: 150).

    Raises:
        ValueError: If combined is not 2D or weights length != 6.
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

    Overlays contours on the edge-filtered image, with a scale bar for biological context.
    Validates inputs to ensure non-empty contours.

    Args:
        image (np.ndarray): Input image (typically edge-filtered).
        contours (list): List of contours (Nx2 arrays from skimage.measure.find_contours).
        output_path (str): Path to save PNG (e.g., results/figures/.../contours.png).
        dpi (int): Resolution (default: 150).

    Raises:
        ValueError: If image is not 2D or contours is empty.
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
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # Red for visibility
    # Add scale bar (assuming 1 pixel ≈ 0.5µm for osteocytes)
    scalebar_length = 50  # 50 pixels ≈ 25µm
    plt.plot([10, 10 + scalebar_length], [10, 10], color='white', linewidth=4)
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

    Shows original, edge-filtered, and labeled mask. Saves two versions: with cell IDs
    (segmentation_labeled.png) and without (segmentation.png). Validates inputs.

    Args:
        image (np.ndarray): Preprocessed image.
        combined (np.ndarray): Edge-filtered image.
        labeled (np.ndarray): Labeled segmentation mask.
        output_path (str): Base path for PNGs (e.g., results/figures/.../segmentation.png).
        dpi (int): Resolution (default: 150 for debugging, 300 for publication).
        fontsize (int): Font size for cell ID annotations (default: 6 to reduce clutter).

    Raises:
        ValueError: If inputs are not 2D or labeled is empty.
    """
    for name, arr in [('image', image), ('combined', combined), ('labeled', labeled)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 2D NumPy array.")
    num_cells = len(np.unique(labeled)) - 1  # Exclude background
    if num_cells == 0:
        logger.warning(f"No cells detected for {output_path}")
        raise ValueError("No cells in labeled mask")

    logger.debug(f"Plotting segmentation: {num_cells} cells, image shape: {image.shape}")
    
    # Plot without cell IDs
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

    # Plot with cell IDs
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

    Generates histograms with KDE and mean lines for intensity, area, dendritic length,
    eccentricity, and solidity. Validates non-empty lists.

    Args:
        image (np.ndarray): Input image for intensity histogram.
        areas (list): List of cell areas.
        dendritic_lengths (list): List of dendritic lengths.
        eccentricities (list): List of eccentricities.
        solidities (list): List of solidities.
        output_path (str): Path to save PNG (e.g., results/figures/.../histograms.png).
        dpi (int): Resolution (default: 150).

    Raises:
        ValueError: If image is not 2D or metric lists are empty.
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

def plot_cellpose_random_walker(image: np.ndarray, cellpose_mask: np.ndarray, rw_prob: np.ndarray, output_path: str, dpi: int = 150):
    """Plot Cellpose and random walker results (optional, unused).

    Shows input image, Cellpose mask, and random walker probability map. Validates inputs.

    Args:
        image (np.ndarray): Input image.
        cellpose_mask (np.ndarray): Cellpose segmentation mask.
        rw_prob (np.ndarray): Random walker probability map.
        output_path (str): Path to save PNG (e.g., results/figures/.../cellpose_random_walker.png).
        dpi (int): Resolution (default: 150).

    Raises:
        ValueError: If inputs are not 2D or empty.
    """
    for name, arr in [('image', image), ('cellpose_mask', cellpose_mask), ('rw_prob', rw_prob)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 2D NumPy array.")
    
    logger.debug(f"Plotting Cellpose/random walker for image shape: {image.shape}")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cellpose_mask, cmap='nipy_spectral')
    plt.title('Cellpose Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(rw_prob, cmap='gray')
    plt.title('Random Walker Probability')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved Cellpose/random walker to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Cellpose/random walker to {output_path}: {e}")
    plt.close()