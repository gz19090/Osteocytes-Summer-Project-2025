# segmentation.py

"""
segmentation.py
---------------
This file provides functions for segmenting osteocyte cells in grayscale images for the osteocyte culture project.
It applies edge detection filters to highlight cell boundaries, combines them using linear regression, and generates
labeled cell masks by filling contours. The segmentation process includes noise reduction, thresholding, and optional
refinement to produce accurate cell boundaries for downstream analysis (e.g., dendrite counting).
Key functions include:
- apply_edge_filters: Combines multiple edge detection filters with learned weights to enhance cell boundaries.
- segment_cells: Segments cells using edge-based contour filling, with options for thresholding and refinement.
Used in the main_workflow.py to process wildtype and mutant osteocyte videos, enabling morphological analysis.
"""

import numpy as np
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace, gaussian
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, closing, disk
from skimage.draw import polygon
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu, threshold_local
from skimage.util import invert
from skimage.measure import label
import logging

# Set up logging to track progress and errors
logger = logging.getLogger(__name__)

def apply_edge_filters(image: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, list]:
    """Apply edge detection filters to highlight cell boundaries and combine them.
    This function uses multiple edge filters (Sobel, Scharr, etc.), reduces noise with a Gaussian filter,
    and combines the results using weights calculated by linear regression to enhance cell edges.
    
    Args:
        image (np.ndarray): 2D grayscale image to process.
        sigma (float): Amount of Gaussian smoothing to reduce noise (default: 1.0).
    
    Returns:
        tuple: (Combined edge image, list of weights for each filter).
    
    Raises:
        ValueError: If the input image is not 2D or is empty.
    """
    # Check if the input image is valid (2D and not empty)
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid input image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")
    
    # Smooth the image to reduce noise
    image = gaussian(image, sigma=sigma, preserve_range=True)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)  # Replace invalid values with 0
    
    # List of edge detection filters to apply
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    filtered_imgs = []
    
    # Apply each edge filter and store the results
    for f in edge_filters:
        try:
            filtered = f(image)  # Apply the filter
            if filtered.shape != image.shape:
                logger.error(f"Filter {f.__name__} produced output shape {filtered.shape}, expected {image.shape}")
                raise ValueError(f"Filter {f.__name__} shape mismatch")
            filtered_imgs.append(filtered.ravel())  # Flatten for regression
            logger.debug(f"Filter {f.__name__} output shape: {filtered.shape}")
        except Exception as e:
            logger.error(f"Error applying filter {f.__name__}: {e}")
            raise
    
    # Combine filter outputs into a matrix
    try:
        X = np.stack(filtered_imgs, axis=1)
        logger.debug(f"Stacked filtered images shape: {X.shape}")
    except Exception as e:
        logger.error(f"Error stacking filtered images: {e}")
        raise
    
    # Use linear regression to find weights for combining filters
    y = image.ravel()
    if len(y) != X.shape[0]:
        logger.error(f"Shape mismatch: y length {len(y)}, X rows {X.shape[0]}")
        raise ValueError("Shape mismatch")
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    weights = reg.coef_  # Get weights for each filter
    logger.info(f"Edge filter weights: {np.round(weights, 3)}")
    
    # Combine filter outputs using the weights
    combined = np.sum([w * f(image) for w, f in zip(weights, edge_filters)], axis=0)
    
    # Normalize the combined image to 0–1 for consistent intensity
    min_val, max_val = combined.min(), combined.max()
    if max_val > min_val:
        combined = (combined - min_val) / (max_val - min_val)
    else:
        logger.warning("Combined image has no range. Returning unnormalized.")
    
    return combined, weights

def segment_cells(image: np.ndarray, min_area: int = 20, use_percentile: bool = False, percentile: float = 90,
                 crop: tuple = None, use_adaptive: bool = False, refine: bool = False, beta: float = 1000) -> tuple:
    """Segment osteocyte cells in an image using edge detection and contour filling.
    This function processes a grayscale image to identify and label individual cells by:
    1) Applying edge filters, 2) Thresholding to create a binary image,
    3) Finding contours, 4) Filling contours to create a mask, and 5) Cleaning up small objects.
    
    Args:
        image (np.ndarray): Preprocessed 2D grayscale image.
        min_area (int): Minimum cell area to keep (default: 20 pixels).
        use_percentile (bool): Use percentile-based thresholding (default: False).
        percentile (float): Percentile value for thresholding (default: 90).
        crop (tuple): Region to crop (y1, y2, x1, x2) or None.
        use_adaptive (bool): Use adaptive thresholding (default: False).
        refine (bool): Use random walker to refine segmentation (default: False).
        beta (float): Random walker stiffness parameter (default: 1000).
    
    Returns:
        tuple: (Labeled cell mask, combined edge image, list of contours).
    
    Raises:
        ValueError: If the image is not 2D, empty, or no cells are detected.
    """
    # Check if the input image is valid
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")
    
    # Crop the image if a crop region is provided
    if crop:
        y1, y2, x1, x2 = crop
        if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
            logger.error(f"Invalid crop {crop} for image shape {image.shape}")
            raise ValueError("Invalid crop region")
        image = image[y1:y2, x1:x2]
    
    # Invert the image (assumes cells are darker than background)
    image = invert(image)
    
    # Apply edge filters to highlight cell boundaries
    combined, weights = apply_edge_filters(image)
    
    # Create a binary image using thresholding
    if use_adaptive:
        thresh = threshold_local(combined, block_size=35, method='gaussian')
        binary = combined > thresh
    else:
        thresh = np.percentile(combined, percentile) if use_percentile else threshold_otsu(combined)
        binary = combined > thresh
    logger.debug(f"Threshold: {thresh if not use_adaptive else 'adaptive'} (use_percentile={use_percentile})")
    
    # Find contours (boundaries) in the binary image
    contours = find_contours(binary, level=0.5)
    logger.debug(f"Found {len(contours)} contours")
    
    # Check if any contours were found
    if len(contours) < 1:
        logger.warning("No contours detected. Check thresholding parameters.")
        raise ValueError("No contours detected")
    
    # Create an empty mask for labeling cells
    segmentation_mask = np.zeros(combined.shape, dtype=np.uint16)
    current_label = 1
    
    # Fill each contour to create a labeled cell region
    for contour in contours:
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=combined.shape)  # Get contour pixels
        existing_labels = segmentation_mask[rr, cc]
        unique_labels = np.unique(existing_labels[existing_labels > 0])
        # Reuse existing label if the contour overlaps a single labeled region
        if len(unique_labels) == 1:
            label_to_use = unique_labels[0]
        else:
            label_to_use = current_label
            current_label += 1
        segmentation_mask[rr, cc] = label_to_use
    
    # Smooth boundaries using morphological closing
    segmentation_mask = closing(segmentation_mask, disk(3))
    
    # Remove small objects (noise) below the minimum area
    cleaned = remove_small_objects(segmentation_mask, min_size=min_area)
    num_cells = len(np.unique(cleaned)) - 1
    logger.debug(f"After filtering: {num_cells} cells")
    
    # Check if any cells remain after filtering
    if num_cells == 0:
        logger.warning("No cells detected after filtering. Adjust min_area or threshold.")
        raise ValueError("No cells detected")
    
    # Refine segmentation using random walker if requested
    if refine:
        from skimage.segmentation import random_walker
        logger.debug("Applying random walker refinement")
        prob_map = random_walker(image, cleaned, beta=beta, mode='cg_mg')
        cleaned = (prob_map > 0.5).astype(np.uint16) * cleaned  # Threshold probability map
        cleaned = label(cleaned)  # Relabel cells
        num_cells = len(np.unique(cleaned)) - 1
        logger.debug(f"After refinement: {num_cells} cells")
    
    return cleaned, combined, contours