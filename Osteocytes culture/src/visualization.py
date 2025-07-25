import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace

def plot_edge_filters(image: np.ndarray, output_path: str):
    """Plot results of edge filters.
    
    Args:
        image (np.ndarray): Input image.
        output_path (str): Path to save figure.
    """
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, f in enumerate(edge_filters):
        filtered = f(image)
        min_val = filtered.min()
        max_val = filtered.max()
        if max_val > min_val:
            filtered = (filtered - min_val) / (max_val - min_val)
        axes[idx].imshow(filtered, cmap='gray')
        axes[idx].set_title(f.__name__)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_image(image: np.ndarray, combined: np.ndarray, weights: list, output_path: str):
    """Plot combined edge-filtered image with weights.
    
    Args:
        image (np.ndarray): Original image.
        combined (np.ndarray): Combined edge-filtered image.
        weights (list): Weights for edge filters.
        output_path (str): Path to save figure.
    """
    plt.figure(figsize=(20, 10))
    plt.imshow(combined, cmap='gray')
    plt.title(f'Best Linear Combination\nWeights: {np.round(weights, 3)}')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_contours(image: np.ndarray, contours: list, output_path: str):
    """Plot contours on the image.
    
    Args:
        image (np.ndarray): Input image.
        contours (list): List of contours.
        output_path (str): Path to save figure.
    """
    plt.figure(figsize=(20, 10))
    plt.imshow(image, cmap='gray')
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_segmentation(image: np.ndarray, combined: np.ndarray, labeled: np.ndarray, output_path: str):
    """Plot segmentation results.
    
    Args:
        image (np.ndarray): Original image.
        combined (np.ndarray): Edge-filtered image.
        labeled (np.ndarray): Labeled cells.
        output_path (str): Path to save figure.
    """
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
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histograms(image: np.ndarray, areas: list, dendritic_lengths: list, eccentricities: list, solidities: list, output_path: str):
    """Plot intensity, area, dendritic length, and eccentricity histograms.
    
    Args:
        image (np.ndarray): Input image.
        areas (list): List of cell areas.
        dendritic_lengths (list): List of dendritic lengths.
        eccentricities (list): List of eccentricities.
        solidities (list): List of solidities.
        output_path (str): Path to save figure.
    """
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.hist(image.ravel(), bins=128)
    plt.title('Histogram of Image Intensity')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 2)
    plt.hist(areas, bins=20)
    plt.title('Cell Area Distribution')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 3)
    plt.hist(dendritic_lengths, bins=20)
    plt.title('Dendritic Length Distribution')
    plt.xlabel('Length (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 4)
    plt.hist(eccentricities, bins=20)
    plt.title('Eccentricity Distribution')
    plt.xlabel('Eccentricity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cellpose_random_walker(image: np.ndarray, cellpose_mask: np.ndarray, rw_prob: np.ndarray, output_path: str):
    """Plot Cellpose and random walker results.
    
    Args:
        image (np.ndarray): Input image.
        cellpose_mask (np.ndarray): Cellpose segmentation mask.
        rw_prob (np.ndarray): Random walker probability map.
        output_path (str): Path to save figure.
    """
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()