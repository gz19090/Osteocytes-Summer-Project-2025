# segmentation.py

import numpy as np
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace, gaussian
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, closing, disk
from skimage.draw import polygon
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu, threshold_local
from skimage.util import invert
from skimage.measure import regionprops
import logging

# Set up logging
logger = logging.getLogger(__name__)

def apply_edge_filters(image: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, list]:
    """Apply linear combination of edge filters with noise reduction.

    Applies multiple edge detection filters, combines them via linear regression, and normalizes output.
    Pre-filters image with Gaussian blur (sigma=1.0) to reduce noise. Validates inputs and logs weights.

    Args:
        image (np.ndarray): Input 2D grayscale image.
        sigma (float): Sigma for Gaussian pre-filtering (default: 1.0, tuned to reduce noise).

    Returns:
        tuple: (Combined edge-filtered image, weights for edge filters).

    Raises:
        ValueError: If image is not 2D or empty.

    Note: Gaussian pre-filter reduces noise amplification in edge detection for osteocyte images.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid input image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")

    # Pre-filter to reduce noise
    image = gaussian(image, sigma=sigma, preserve_range=True)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    filtered_imgs = []
    for f in edge_filters:
        try:
            filtered = f(image)
            if filtered.shape != image.shape:
                logger.error(f"Filter {f.__name__} produced output shape {filtered.shape}, expected {image.shape}")
                raise ValueError(f"Filter {f.__name__} shape mismatch")
            filtered_imgs.append(filtered.ravel())
            logger.debug(f"Filter {f.__name__} output shape: {filtered.shape}")
        except Exception as e:
            logger.error(f"Error applying filter {f.__name__}: {e}")
            raise

    try:
        X = np.stack(filtered_imgs, axis=1)
        logger.debug(f"Stacked filtered images shape: {X.shape}")
    except Exception as e:
        logger.error(f"Error stacking filtered images: {e}")
        raise

    y = image.ravel()
    if len(y) != X.shape[0]:
        logger.error(f"Shape mismatch: y length {len(y)}, X rows {X.shape[0]}")
        raise ValueError("Shape mismatch")

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    weights = reg.coef_
    logger.info(f"Edge filter weights: {np.round(weights, 3)}")

    combined = np.sum([w * f(image) for w, f in zip(weights, edge_filters)], axis=0)
    min_val, max_val = combined.min(), combined.max()
    if max_val > min_val:
        combined = (combined - min_val) / (max_val - min_val)
    else:
        logger.warning("Combined image has no range. Returning unnormalized.")
    
    return combined, weights

# Check different percentiles
def segment_cells(image: np.ndarray, min_area: int = 50, use_percentile: bool = False, percentile: float = 94,
                 crop: tuple = None, use_adaptive: bool = False, refine: bool = False, beta: float = 1000) -> tuple:
    """Segment osteocyte cells using contour-based approach with noise reduction.

    Inverts image, applies edge filters, thresholds (percentile, Otsu, or adaptive), finds contours,
    fills them, applies morphological closing, and filters small objects. Optional random walker
    refinement reduces noise. Validates inputs and cell count.

    Args:
        image (np.ndarray): Preprocessed 2D grayscale image.
        min_area (int): Min object area (default: 50, tuned to filter noise in osteocytes).
        use_percentile (bool): Use percentile thresholding (default: False, Otsu).
        percentile (float): Percentile for thresholding (default: 94, stricter for noise).
        crop (tuple): Crop region (y1, y2, x1, x2) or None.
        use_adaptive (bool): Use adaptive thresholding for variable lighting (default: False).
        refine (bool): Apply random walker refinement (default: False).
        beta (float): Random walker stiffness (default: 1000, smoother boundaries).

    Returns:
        tuple: (Labeled segmentation mask, combined edge image, contours).

    Raises:
        ValueError: If image is not 2D, empty, or no cells detected.

    Note: Closing (disk radius=3) and min_area=50 reduce noise; percentile=85 tuned for mutant videos.
    Reference: Inspired by scikit-image segmentation workflows for biological imaging.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        logger.error(f"Invalid image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input must be a non-empty 2D NumPy array.")

    if crop:
        y1, y2, x1, x2 = crop
        if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
            logger.error(f"Invalid crop {crop} for image shape {image.shape}")
            raise ValueError("Invalid crop region")
        image = image[y1:y2, x1:x2]

    image = invert(image)  # Assumes cells darker than background (phase-contrast)
    combined, weights = apply_edge_filters(image)

    if use_adaptive:
        thresh = threshold_local(combined, block_size=35, method='gaussian')
        binary = combined > thresh
    else:
        thresh = np.percentile(combined, percentile) if use_percentile else threshold_otsu(combined)
        binary = combined > thresh
    logger.debug(f"Threshold: {thresh if not use_adaptive else 'adaptive'} (use_percentile={use_percentile})")

    contours = find_contours(binary, level=0.5)
    logger.debug(f"Found {len(contours)} contours")
    if len(contours) < 1:
        logger.warning("No contours detected. Check thresholding parameters.")
        raise ValueError("No contours detected")

    segmentation_mask = np.zeros(combined.shape, dtype=np.uint16)
    current_label = 1
    for contour in contours:
        rr, cc = polygon(contour[:, 0], contour[:, 1], shape=combined.shape)
        existing_labels = segmentation_mask[rr, cc]
        unique_labels = np.unique(existing_labels[existing_labels > 0])
        if len(unique_labels) == 1:
            label_to_use = unique_labels[0]
        else:
            label_to_use = current_label
            current_label += 1
        segmentation_mask[rr, cc] = label_to_use

    # Apply morphological closing to smooth boundaries
    segmentation_mask = closing(segmentation_mask, disk(3))
    cleaned = remove_small_objects(segmentation_mask, min_size=min_area)
    num_cells = len(np.unique(cleaned)) - 1
    logger.debug(f"After filtering: {num_cells} cells")
    if num_cells == 0:
        logger.warning("No cells detected after filtering. Adjust min_area or threshold.")
        raise ValueError("No cells detected")

    if refine:
        logger.debug("Applying random walker refinement")
        prob_map = refine_with_random_walker(image, cleaned, beta=beta)
        cleaned = (prob_map > 0.5).astype(np.uint16) * cleaned  # Threshold probability map
        cleaned = label(cleaned)  # Relabel after refinement
        num_cells = len(np.unique(cleaned)) - 1
        logger.debug(f"After refinement: {num_cells} cells")

    return cleaned, combined, contours