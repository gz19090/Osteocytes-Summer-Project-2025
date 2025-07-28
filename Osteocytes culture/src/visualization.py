# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)

def plot_edge_filters(image: np.ndarray, output_path: str):
    """
    Plot results of edge filters.
    
    This function applies six edge detection filters (Sobel, Scharr, Prewitt, Roberts, Farid, Laplace)
    to the input image, normalizes each output, and displays them in a 2x3 subplot grid. Useful for
    visualizing individual edge filter contributions before combination.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    # Create a 2x3 subplot grid for displaying the filtered images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, f in enumerate(edge_filters):
        # Apply the edge filter to the image
        filtered = f(image)
        min_val = filtered.min()
        max_val = filtered.max()
        # Normalize the filtered image if it has a non-zero range to [0, 1]
        if max_val > min_val:
            filtered = (filtered - min_val) / (max_val - min_val)
        # Display the normalized filtered image in grayscale
        axes[idx].imshow(filtered, cmap='gray')
        axes[idx].set_title(f.__name__)
        axes[idx].axis('off')
    
    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_image(image: np.ndarray, combined: np.ndarray, weights: list, output_path: str):
    """Plot combined edge-filtered image with weights.
    
    This function displays the combined edge-filtered image (linear combination of edge filters)
    and includes the computed weights in the title. Useful for inspecting the optimized edge enhancement.

    Args:
        image (np.ndarray): Original image (not used for plotting but included for consistency).
        combined (np.ndarray): Combined edge-filtered image.
        weights (list): Weights for edge filters (from linear regression).
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    # Create a large figure for the combined image
    plt.figure(figsize=(20, 10))
    # Display the combined image in grayscale
    plt.imshow(combined, cmap='gray')
    # Set title with rounded weights for readability
    plt.title(f'Best Linear Combination\nWeights: {np.round(weights, 3)}')
    plt.axis('off')
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_contours(image: np.ndarray, contours: list, output_path: str):
    """Plot contours on the image.
    
    This function overlays detected contours on the input image (typically the edge-filtered image).
    Useful for visualizing cell boundaries before labeling.

    Args:
        image (np.ndarray): Input image to overlay contours on.
        contours (list): List of contours (each a Nx2 array of coordinates).
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    # Create a large figure for the contour overlay
    plt.figure(figsize=(20, 10))
    # Display the input image in grayscale
    plt.imshow(image, cmap='gray')
    # Plot each contour as a line with linewidth=2
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.axis('off')
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_segmentation(image: np.ndarray, combined: np.ndarray, labeled: np.ndarray, output_path: str):
    """Plot segmentation results.
    
    This function creates a side-by-side comparison of the original image, edge-filtered image,
    and labeled segmentation mask. The labeled plot includes the number of detected cells.

    Args:
        image (np.ndarray): Original (preprocessed) image.
        combined (np.ndarray): Edge-filtered image.
        labeled (np.ndarray): Labeled cells (segmentation mask).
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    # Create a 1x3 subplot grid for comparison
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
    plt.title(f'Labeled Cells ({np.max(labeled)})')
    plt.axis('off')
    
    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histograms(image: np.ndarray, areas: list, dendritic_lengths: list, eccentricities: list, solidities: list, output_path: str):
    """Plot intensity, area, dendritic length, and eccentricity histograms.
    
    This function generates histograms for image intensity and selected cell metrics (area, dendritic length,
    eccentricity, solidity). Useful for inspecting distributions and identifying outliers in metrics.

    Args:
        image (np.ndarray): Input image for intensity histogram.
        areas (list): List of cell areas.
        dendritic_lengths (list): List of dendritic lengths.
        eccentricities (list): List of eccentricities.
        solidities (list): List of solidities.
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    # Create a 1x4 subplot grid for multiple histograms
    plt.figure(figsize=(16, 4))
    
    # Intensity histogram
    plt.subplot(1, 4, 1)
    plt.hist(image.ravel(), bins=128)
    plt.title('Histogram of Image Intensity')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    
    # Cell area distribution
    plt.subplot(1, 4, 2)
    plt.hist(areas, bins=20)
    plt.title('Cell Area Distribution')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    
    # Dendritic length distribution
    plt.subplot(1, 4, 3)
    plt.hist(dendritic_lengths, bins=20)
    plt.title('Dendritic Length Distribution')
    plt.xlabel('Length (pixels)')
    plt.ylabel('Count')
    
    # Eccentricity distribution
    plt.subplot(1, 4, 4)
    plt.hist(eccentricities, bins=20)
    plt.title('Eccentricity Distribution')
    plt.xlabel('Eccentricity')
    plt.ylabel('Count')
    
    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cellpose_random_walker(image: np.ndarray, cellpose_mask: np.ndarray, rw_prob: np.ndarray, output_path: str):
    """Plot Cellpose and random walker results.
    
    This function creates a side-by-side comparison of the input image, Cellpose segmentation mask,
    and random walker probability map. This is optional and used only if Cellpose is enabled.

    Args:
        image (np.ndarray): Input image.
        cellpose_mask (np.ndarray): Cellpose segmentation mask.
        rw_prob (np.ndarray): Random walker probability map.
        output_path (str): Path to save the figure as a high-resolution PNG.
    """
    # Create a 1x3 subplot grid for comparison
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
    
    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()
    # Save the figure with high resolution and close to free memory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()