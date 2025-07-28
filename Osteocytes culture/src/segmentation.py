# segmentation.py

import numpy as np
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects
from skimage.draw import polygon
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

try:
    from cellpose import models
    import torch
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
from skimage.segmentation import random_walker
from skimage.measure import regionprops, label
import logging

# Set up logging for debugging and error reporting in segmentation processes
logger = logging.getLogger(__name__)

def apply_edge_filters(image: np.ndarray) -> tuple[np.ndarray, list]:
    """Apply linear combination of edge filters.
    
    This function applies multiple edge detection filters to the input image, flattens their outputs,
    stacks them into a feature matrix, and uses linear regression to compute optimal weights for
    combining them into a single enhanced edge image. It includes input validation, NaN handling,
    and logging for debugging shape issues.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
    
    Returns:
        tuple: (Combined edge-filtered image, weights for edge filters).
    """
    # Validate input image type and dimensions
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        logger.error(f"Invalid input image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input image must be a 2D NumPy array.")
    
    # Handle NaN or Inf values by replacing with zeros to prevent errors in filtering
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        logger.warning("Input image contains NaN or Inf values. Replacing with zeros.")
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # List of edge detection filters to apply
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    filtered_imgs = []
    for f in edge_filters:
        try:
            # Apply the filter and flatten the result for stacking
            filtered = f(image)
            # Check if filter output matches input shape
            if filtered.shape != image.shape:
                logger.error(f"Filter {f.__name__} produced output shape {filtered.shape}, expected {image.shape}")
                raise ValueError(f"Filter {f.__name__} shape mismatch")
            filtered_imgs.append(filtered.ravel())
            logger.debug(f"Filter {f.__name__} output shape: {filtered.shape}, flattened length: {len(filtered.ravel())}")
        except Exception as e:
            logger.error(f"Error applying filter {f.__name__}: {e}")
            raise
    
    try:
        # Stack flattened filtered images into a feature matrix X
        X = np.stack(filtered_imgs, axis=1)
        logger.debug(f"Stacked filtered images shape: {X.shape}")
    except Exception as e:
        logger.error(f"Error stacking filtered images: {e}")
        raise
    
    # Flatten the input image as the target y for regression
    y = image.ravel()
    # Check for shape mismatch between y and X
    if len(y) != X.shape[0]:
        logger.error(f"Shape mismatch: y length {len(y)}, X rows {X.shape[0]}")
        raise ValueError("Shape mismatch between input image and filtered images")
    
    # Fit linear regression without intercept to compute weights
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    weights = reg.coef_
    # Combine filters using computed weights
    combined = np.sum([w * f(image) for w, f in zip(weights, edge_filters)], axis=0)
    
    # Normalize combined image to [0, 1] if it has a non-zero range
    min_val = combined.min()
    max_val = combined.max()
    if max_val > min_val:
        combined = (combined - min_val) / (max_val - min_val)
    else:
        logger.warning("Combined image has no range (max_val == min_val). Returning unnormalized.")
    
    return combined, weights

# Updated to include segmentation percentile
def segment_cells(image: np.ndarray, min_area: int = 10, use_percentile: bool = False, 
                 percentile: float = 87, crop: tuple = None) -> tuple:
    """Segment cells using contour-based approach.
    
    This function inverts the image, applies edge filtering, thresholds (percentile or Otsu),
    finds contours, fills them to create a mask, and filters small objects. It includes logging
    for threshold values, number of contours, and final objects. Optional crop is applied first.

    Args:
        image (np.ndarray): Preprocessed image (2D grayscale array).
        min_area (int): Minimum area for objects (to filter small noise).
        use_percentile (bool): Use percentile thresholding instead of Otsu.
        percentile (float): Percentile for thresholding (if use_percentile=True).
        crop (tuple): Crop region (y1, y2, x1, x2) or None for full image.
    
    Returns:
        tuple: (Labeled segmentation mask, combined edge image, contours).
    """
    # Apply crop if specified
    if crop:
        y1, y2, x1, x2 = crop
        image = image[y1:y2, x1:x2]
    
    # Invert the image for edge detection (assumes cells are darker than background)
    image = invert(image)
    # Apply edge filters to enhance boundaries
    combined, weights = apply_edge_filters(image)
    
    # Compute threshold based on user choice
    if use_percentile:
        thresh = np.percentile(combined, percentile)
        logger.debug(f"Using percentile threshold: {thresh} (percentile={percentile})")
    else:
        thresh = threshold_otsu(combined)
        logger.debug(f"Using Otsu threshold: {thresh}")
    
    # Find contours in the thresholded edge image
    contours = find_contours(combined, thresh)
    logger.debug(f"Found {len(contours)} contours")
    
    # Create a blank mask for labeling regions
    segmentation_mask = np.zeros(combined.shape, dtype=np.uint16)
    current_label = 1
    for contour in contours:
        # Fill the contour region using polygon fill
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=combined.shape)
        existing_labels = segmentation_mask[rr, cc]
        unique_labels = np.unique(existing_labels[existing_labels > 0])
        # Handle overlapping contours by assigning existing label if single overlap
        if len(unique_labels) == 1:
            label_to_use = unique_labels[0]
        else:
            label_to_use = current_label
            current_label += 1
        segmentation_mask[rr, cc] = label_to_use
    
    # Remove small objects (noise) from the mask
    cleaned = remove_small_objects(segmentation_mask, min_size=min_area)
    logger.debug(f"After filtering, {len(np.unique(cleaned)) - 1} objects remain")
    return cleaned, combined, contours

def segment_cells_cellpose(image: np.ndarray) -> np.ndarray:
    """Segment cells using Cellpose 4.0.6+ (optional).
    
    This function converts the image to float32, attempts MPS GPU acceleration (falls back to CPU),
    applies the Scharr filter as preprocessing, and runs Cellpose evaluation for segmentation.
    It is disabled by default due to compatibility issues but can be uncommented if enabled.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
    
    Returns:
        np.ndarray: Labeled segmentation mask.
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose is not installed. Run 'pip install cellpose'.")
    
    # Convert to float32 to avoid data type issues
    image = image.astype(np.float32)
    try:
        # Try MPS for GPU acceleration if available
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = models.CellposeModel(gpu=(device.type == "mps"), device=device)
        masks, _, _, _ = model.eval([scharr(image)], diameter=None, channels=[0, 0])
    except Exception as e:
        print(f"MPS failed: {e}. Falling back to CPU.")
        device = torch.device('cpu')  # Corrected typo from 'torch.device个体'
        model = models.CellposeModel(gpu=False, device=device)
        masks, _, _, _ = model.eval([scharr(image)], diameter=None, channels=[0, 0])
    
    return masks[0]

def refine_with_random_walker(image: np.ndarray, labeled: np.ndarray, beta: float = 1000) -> np.ndarray:
    """Refine segmentation using random walker.
    
    This function creates markers from centroids (cells) and background, then applies the random walker
    algorithm to refine the segmentation based on intensity differences. It returns a probability map
    for the cell class.

    Args:
        image (np.ndarray): Input image (inverted for better contrast).
        labeled (np.ndarray): Initial labeled mask.
        beta (float): Random walker stiffness parameter (higher beta = smoother boundaries).
    
    Returns:
        np.ndarray: Refined probability map for the cell class.
    """
    # Extract centroids from labeled regions
    regions = regionprops(labeled)
    centres = [region.centroid for region in regions]
    centres_img = np.zeros_like(labeled)
    for i, (y, x) in enumerate(centres, 1):
        cy, cx = int(round(y)), int(round(x))
        # Ensure centroid coordinates are within bounds
        if 0 <= cy < centres_img.shape[0] and 0 <= cx < centres_img.shape[1]:
            centres_img[cy, cx] = i
    
    # Define confident markers for cells and background
    confident_cell = centres_img > 0
    confident_bkg = labeled == 0
    markers = np.zeros_like(labeled)
    markers[confident_bkg] = 1
    markers[confident_cell] = centres_img[confident_cell]
    
    # Run random walker with 'bf' mode and return full probabilities
    pb = random_walker(image, markers, beta=beta, mode='bf', return_full_prob=True)
    return pb[1]