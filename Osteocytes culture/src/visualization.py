# visualization.py

"""
visualization.py
----------------
This module provides functions to visualize osteocyte cell image analysis results for the osteocyte culture project.
It generates plots to debug and validate cell segmentation, edge detection, and dendrite counting, including edge filter outputs,
segmentation overlays, histograms of cell metrics, and skeleton overlays for full and dendritic structures.
Key functions include:
- plot_edge_filters: Visualizes outputs of edge detection filters for debugging cell boundaries.
- plot_combined_image: Shows the combined edge-filtered image with weights.
- plot_contours: Overlays cell boundary contours on the image with a scale bar.
- plot_segmentation: Displays original, edge-filtered, and labeled cell images, with optional cell ID annotations.
- plot_histograms: Plots distributions of image intensity and cell metrics (area, dendrite count, eccentricity).
- plot_skeleton_overlays: Creates overlays of full and dendrite-only skeletons for each cell.
Used in the main_workflow.py to generate figures for wildtype and mutant osteocyte videos, aiding in the analysis of cell morphology.
"""

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

# Set up logging to track progress and errors
logger = logging.getLogger(__name__)

def plot_edge_filters(image: np.ndarray, output_path: str, dpi: int = 150):
    """Create and save a plot showing different edge detection filters applied to an image.
    This function applies six edge filters (Sobel, Scharr, etc.) to highlight cell boundaries
    and saves them as a single image for debugging.
    
    Args:
        image (np.ndarray): 2D grayscale image to process.
        output_path (str): File path to save the plot as a PNG.
        dpi (int): Image resolution (default: 150).
    """
    # Check if the input image is valid (2D and not empty)
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")
    
    logger.debug(f"Plotting edge filters for image shape: {image.shape}")
    
    # List of edge detection filters to apply
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()  # Flatten the grid for easier access
    
    # Apply each filter and plot the result
    for idx, f in enumerate(edge_filters):
        filtered = f(image)  # Apply the filter
        min_val, max_val = filtered.min(), filtered.max()
        # Normalize the filtered image to 0–1 for consistent display
        if max_val > min_val:
            filtered = (filtered - min_val) / (max_val - min_val)
        axes[idx].imshow(filtered, cmap='gray')  # Show as grayscale
        axes[idx].set_title(f.__name__)  # Set filter name as title
        axes[idx].axis('off')  # Hide axes
        logger.debug(f"Filter {f.__name__} range: {min_val:.2f}, {max_val:.2f}")
    
    plt.tight_layout()  # Adjust spacing for clarity
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')  # Save the plot
        logger.info(f"Saved edge filters to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save edge filters to {output_path}: {e}")
    plt.close()  # Close the plot to free memory

def plot_combined_image(image: np.ndarray, combined: np.ndarray, weights: list, output_path: str, dpi: int = 150):
    """Create and save a plot of the combined edge-filtered image with filter weights.
    This function shows the result of combining edge filters with specific weights.
    
    Args:
        image (np.ndarray): Original image (not used, kept for consistent function calls).
        combined (np.ndarray): Combined edge-filtered image.
        weights (list): Weights used for combining edge filters (must have 6 values).
        output_path (str): File path to save the plot as a PNG.
        dpi (int): Image resolution (default: 150).
    """
    # Check if the combined image is valid
    if not isinstance(combined, np.ndarray) or combined.ndim != 2 or combined.size == 0:
        raise ValueError("Combined image must be a non-empty 2D NumPy array.")
    if len(weights) != 6:
        raise ValueError(f"Expected 6 weights, got {len(weights)}")
    
    logger.debug(f"Plotting combined image shape: {combined.shape}, weights: {np.round(weights, 3)}")
    
    # Create a plot
    plt.figure(figsize=(20, 10))
    plt.imshow(combined, cmap='gray')  # Show combined image in grayscale
    
    # Create title with filter names and their weights
    filter_names = ['Sobel', 'Scharr', 'Prewitt', 'Roberts', 'Farid', 'Laplace']
    title = f'Best Linear Combination\nWeights: {", ".join(f"{n}={w:.3f}" for n, w in zip(filter_names, weights))}'
    plt.title(title)
    plt.axis('off')  # Hide axes
    
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')  # Save the plot
        logger.info(f"Saved combined image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save combined image to {output_path}: {e}")
    plt.close()  # Close the plot to free memory

def plot_contours(image: np.ndarray, contours: list, output_path: str, dpi: int = 150):
    """Create and save a plot showing cell boundaries (contours) on an image.
    This function overlays red contours on the image and adds a scale bar (25 µm).
    
    Args:
        image (np.ndarray): Input image (usually edge-filtered).
        contours (list): List of contours (arrays of boundary points from skimage).
        output_path (str): File path to save the plot as a PNG.
        dpi (int): Image resolution (default: 150).
    """
    # Check if the input image and contours are valid
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D NumPy array.")
    if not contours:
        logger.warning(f"No contours provided for {output_path}")
        raise ValueError("Contours list is empty")
    
    logger.debug(f"Plotting {len(contours)} contours on image shape: {image.shape}")
    
    # Create a plot
    plt.figure(figsize=(20, 10))
    plt.imshow(image, cmap='gray')  # Show image in grayscale
    
    # Draw each contour in red
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    
    # Add a scale bar (25 µm)
    plt.plot([10, 60], [10, 10], color='white', linewidth=4)
    plt.text(10, 20, '25 µm', color='white', fontsize=12)
    plt.axis('off')  # Hide axes
    
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')  # Save the plot
        logger.info(f"Saved contours to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save contours to {output_path}: {e}")
    plt.close()  # Close the plot to free memory

def plot_segmentation(image: np.ndarray, combined: np.ndarray, labeled: np.ndarray, output_path: str, dpi: int = 150, fontsize: int = 6):
    """Create and save plots showing the original, edge-filtered, and labeled cell images.
    This function makes two plots: one with just the labeled cells and another with cell IDs labeled.
    
    Args:
        image (np.ndarray): Original preprocessed image.
        combined (np.ndarray): Edge-filtered image.
        labeled (np.ndarray): Labeled mask where each cell has a unique number.
        output_path (str): File path to save the plot without IDs.
        dpi (int): Image resolution (default: 150).
        fontsize (int): Font size for cell ID labels (default: 6).
    """
    # Check if inputs are valid
    for name, arr in [('image', image), ('combined', combined), ('labeled', labeled)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty 2D NumPy array.")
    
    # Count number of cells (excluding background)
    num_cells = len(np.unique(labeled)) - 1
    if num_cells == 0:
        logger.warning(f"No cells detected for {output_path}")
        raise ValueError("No cells in labeled mask")
    
    logger.debug(f"Plotting segmentation: {num_cells} cells, image shape: {image.shape}")
    
    # Plot 1: Original, edge-filtered, and labeled images (no IDs)
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
    plt.imshow(labeled, cmap='nipy_spectral')  # Colorful map for labeled cells
    plt.title(f'Labeled Cells ({num_cells})')
    plt.axis('off')
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved segmentation (without IDs) to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save segmentation to {output_path}: {e}")
    plt.close()
    
    # Plot 2: Same as above but with cell IDs labeled
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
    # Add cell ID labels at the center of each cell
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
    """Create and save histograms of image intensity and cell metrics.
    This function plots histograms for intensity, area, dendrite count, and eccentricity,
    with mean lines for reference.
    
    Args:
        image (np.ndarray): 2D grayscale image.
        areas (list): List of cell areas.
        dendrite_counts (list): List of dendrite counts per cell.
        eccentricities (list): List of cell eccentricities.
        solidities (list): List of cell solidities.
        output_path (str): File path to save the plot as a PNG.
        dpi (int): Image resolution (default: 150).
    """
    # Check if inputs are valid
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D NumPy array.")
    for name, lst in [('areas', areas), ('dendrite_counts', dendrite_counts),
                      ('eccentricities', eccentricities), ('solidities', solidities)]:
        if lst is None or len(lst) == 0:
            logger.warning(f"Empty {name} list for {output_path}")
            raise ValueError(f"{name} list is empty")
    
    logger.debug(f"Plotting histograms: {len(areas)} cells, image shape: {image.shape}")
    
    # Set plot style for clarity
    sns.set_style('whitegrid')
    plt.figure(figsize=(16, 4))
    
    # Histogram 1: Image intensity
    plt.subplot(1, 4, 1)
    sns.histplot(image.ravel(), bins=128, kde=True)  # Plot intensity distribution
    plt.axvline(np.mean(image.ravel()), color='red', linestyle='--', label='Mean')
    plt.title('Image Intensity')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.legend()
    
    # Histogram 2: Cell area
    plt.subplot(1, 4, 2)
    sns.histplot(areas, bins=min(20, len(areas)//2 or 1), kde=True)
    plt.axvline(np.mean(areas), color='red', linestyle='--', label='Mean')
    plt.title('Cell Area')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    plt.legend()
    
    # Histogram 3: Dendrite count (discrete)
    plt.subplot(1, 4, 3)
    dmin, dmax = int(np.min(dendrite_counts)), int(np.max(dendrite_counts))
    bins = np.arange(dmin, dmax + 2) - 0.5  # Integer bins for discrete counts
    sns.histplot(dendrite_counts, bins=bins, discrete=True)
    plt.axvline(np.mean(dendrite_counts), color='red', linestyle='--', label='Mean')
    plt.title('Dendrite Count')
    plt.xlabel('# Dendrites')
    plt.ylabel('Cells')
    plt.legend()
    
    # Histogram 4: Eccentricity
    plt.subplot(1, 4, 4)
    sns.histplot(eccentricities, bins=min(20, len(eccentricities)//2 or 1), kde=True)
    plt.axvline(np.mean(eccentricities), color='red', linestyle='--', label='Mean')
    plt.title('Eccentricity')
    plt.xlabel('Eccentricity')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')  # Save the plot
        logger.info(f"Saved histograms to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save histograms to {output_path}: {e}")
    plt.close()

def _protrusion_skeleton(cell_mask: np.ndarray, soma_frac: float = 0.18, min_component_size: int = 8) -> np.ndarray:
    """Create a skeleton of a cell's dendrites by removing the main cell body.
    This function isolates thin processes (dendrites) by removing the cell body
    using morphological opening and skeletonizing the remaining parts.
    
    Args:
        cell_mask (np.ndarray): Binary mask of a single cell (bool).
        soma_frac (float): Fraction of cell diameter to estimate cell body size (default: 0.18).
        min_component_size (int): Minimum size of protrusions to keep (default: 8 pixels).
    
    Returns:
        np.ndarray: Skeleton of dendrites (2D, bool).
    """
    # Convert mask to boolean
    cell = cell_mask.astype(bool, copy=False)
    
    # Return empty skeleton if the mask is empty
    if cell.sum() == 0:
        return np.zeros_like(cell, dtype=bool)
    
    # Calculate cell area
    area = int(cell.sum())
    
    # Estimate cell body radius from area
    equiv_d = 2.0 * np.sqrt(area / np.pi)
    r = max(2, int(round(soma_frac * equiv_d)))  # Ensure radius is at least 2 pixels
    
    # Remove cell body using morphological opening
    core = opening(cell, disk(r))
    
    # Isolate dendrites by subtracting the core
    protrusions = cell & ~core
    
    # Clean up small noise in protrusions
    if protrusions.any():
        protrusions = remove_small_objects(protrusions, min_size=min_component_size)
    
    # Skeletonize the dendrites
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
        fill_color: str = "#FFA726",  # orange
        outline_color: str = "saddlebrown",
):
    """Create and save plots showing cell skeletons overlaid on a grayscale background.
    For each cell, two plots are made: one with the full skeleton (light gray) and one
    with only the dendrite skeleton (white), showing the counted dendrites.
    
    Args:
        labeled (np.ndarray): Labeled mask where each cell has a unique number.
        image_processed (np.ndarray): Grayscale background image (e.g., edge-filtered).
        cell_metrics (pd.DataFrame): Table with cell metrics, including dendrite counts.
        output_dir (str): Folder to save the plots.
        image_original (np.ndarray): Optional original image (not used).
        soma_frac (float): Fraction of cell diameter for cell body estimation (default: 0.18).
        min_component_size (int): Minimum size of protrusions (default: 8 pixels).
        min_len_px (int): Minimum pixels for a dendrite branch (default: 5).
        alpha_mask (float): Transparency for cell fill (default: 0.38).
        fill_color (str): Color for cell fill (default: orange).
        outline_color (str): Color for cell outline (default: saddlebrown).
    """
    # Check if inputs are valid
    if not isinstance(labeled, np.ndarray) or labeled.ndim != 2 or labeled.size == 0:
        raise ValueError("Labeled mask must be a non-empty 2D NumPy array.")
    if not isinstance(image_processed, np.ndarray) or image_processed.ndim != 2 or image_processed.size == 0:
        raise ValueError("image_processed must be a non-empty 2D NumPy array.")
    if 'dendrite_count' not in cell_metrics.columns:
        raise ValueError("cell_metrics must include 'dendrite_count'.")
    
    # Get cell IDs from metrics
    metrics_idx = (cell_metrics['cell_id'].values if 'cell_id' in cell_metrics.columns
                   else cell_metrics.index.values).astype(int)
    
    # Ensure labeled mask has sequential IDs
    labeled, _, _ = relabel_sequential(labeled)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Map cell IDs to dendrite counts
    dendrite_map = {int(cid): int(cnt) for cid, cnt in zip(metrics_idx, cell_metrics['dendrite_count'].values)}
    
    # Define colors for plotting
    fill_cmap = ListedColormap([fill_color])  # Color for cell fill
    outline_cmap = ListedColormap([outline_color])  # Color for cell outline
    full_cmap = ListedColormap(['#DCDCDC'])  # Light gray for full skeleton
    prot_cmap = ListedColormap(['#FFFFFF'])  # White for dendrite skeleton
    
    # Get regions (cells) from the labeled mask
    cell_slices = ndimage.find_objects(labeled)
    
    # Process each cell
    for i, s in enumerate(cell_slices, start=1):
        if s is None:
            continue
        # Get the region for the current cell
        rsl, csl = s
        r0, r1 = rsl.start, rsl.stop
        c0, c1 = csl.start, csl.stop
        pad = 5  # Add padding around the cell
        r0p = max(0, r0 - pad)
        r1p = min(labeled.shape[0], r1 + pad)
        c0p = max(0, c0 - pad)
        c1p = min(labeled.shape[1], c1 + pad)
        
        # Extract cell mask and background
        lab_view = labeled[r0p:r1p, c0p:c1p]
        bg_view = image_processed[r0p:r1p, c0p:c1p]
        cell_mask = (lab_view == i)
        
        # Create full and dendrite skeletons
        skel_full = skeletonize(cell_mask)
        skel_prot = _protrusion_skeleton(cell_mask, soma_frac=soma_frac, min_component_size=min_component_size)
        if skel_prot.sum() < min_len_px:
            skel_prot[:] = False  # Clear small skeletons
        
        # Create boundaries and masked arrays for plotting
        outline = find_boundaries(cell_mask, mode='outer')
        mask_ma = np.ma.masked_where(~cell_mask, cell_mask)
        outl_ma = np.ma.masked_where(~outline, outline)
        full_ma = np.ma.masked_where(~skel_full, skel_full)
        prot_ma = np.ma.masked_where(~skel_prot, skel_prot)
        
        # Create plot with two panels
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
        panels = [
            ("FULL skeleton", full_ma, full_cmap),
            (f"PROTRUSION skeleton (counted: {dendrite_map.get(i, 0)})", prot_ma, prot_cmap),
        ]
        
        # Plot each panel
        for ax, (subtitle, skel_ma, skel_cmap) in zip(axes, panels):
            ax.imshow(bg_view, cmap="gray", interpolation="nearest")  # Show grayscale background
            ax.imshow(mask_ma, cmap=fill_cmap, alpha=alpha_mask, interpolation="nearest", vmin=0, vmax=1)  # Cell fill
            ax.imshow(outl_ma, cmap=outline_cmap, interpolation="nearest", vmin=0, vmax=1)  # Cell outline
            ax.imshow(skel_ma, cmap=skel_cmap, interpolation="nearest", vmin=0, vmax=1)  # Skeleton overlay
            ax.set_title(f"Cell {i} | {subtitle}")
            ax.axis("off")
        
        # Save the plot
        out_path = Path(output_dir) / f"skeleton_overlay_label_{i}.png"
        try:
            plt.savefig(out_path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)  # Always close to free memory