import numpy as np
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects
from skimage.draw import polygon
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu
from skimage.util import invert
try:
    from cellpose import models
    import torch
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
from skimage.segmentation import random_walker
from skimage.measure import regionprops, label
import logging

# Set up logging
logger = logging.getLogger(__name__)

def apply_edge_filters(image: np.ndarray) -> tuple[np.ndarray, list]:
    """Apply linear combination of edge filters.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        tuple: (Combined edge-filtered image, weights for edge filters).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        logger.error(f"Invalid input image shape: {image.shape if isinstance(image, np.ndarray) else type(image)}")
        raise ValueError("Input image must be a 2D NumPy array.")
    
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        logger.warning("Input image contains NaN or Inf values. Replacing with zeros.")
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
            logger.debug(f"Filter {f.__name__} output shape: {filtered.shape}, flattened length: {len(filtered.ravel())}")
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
        raise ValueError("Shape mismatch between input image and filtered images")
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    weights = reg.coef_
    combined = np.sum([w * f(image) for w, f in zip(weights, edge_filters)], axis=0)
    
    min_val = combined.min()
    max_val = combined.max()
    if max_val > min_val:
        combined = (combined - min_val) / (max_val - min_val)
    else:
        logger.warning("Combined image has no range (max_val == min_val). Returning unnormalized.")
    
    return combined, weights

# def segment_cells(image: np.ndarray, min_area: int = 10, use_percentile: bool = False, 
#                 percentile: float = 87, crop: tuple = None) -> tuple:
#     """Segment cells using contour-based approach.
    
#     Args:
#         image (np.ndarray): Preprocessed image.
#         min_area (int): Minimum area for objects.
#         use_percentile (bool): Use percentile thresholding instead of Otsu.
#         percentile (float): Percentile for thresholding (if use_percentile=True).
#         crop (tuple): Crop region (y1, y2, x1, x2) or None for full image.
    
#     Returns:
#         tuple: (Labeled segmentation mask, combined edge image, contours).
#     """
#     if crop:
#         y1, y2, x1, x2 = crop
#         image = image[y1:y2, x1:x2]
    
#     image = invert(image)
#     combined, weights = apply_edge_filters(image)
    
#     if use_percentile:
#         thresh = np.percentile(combined, percentile)
#     else:
#         thresh = threshold_otsu(combined)
    
#     contours = find_contours(combined, thresh)
    
#     segmentation_mask = np.zeros(combined.shape, dtype=np.uint16)
#     current_label = 1
#     for contour in contours:
#         rr, cc = polygon(contour[:, 0], contour[:, 1], shape=combined.shape)
#         existing_labels = segmentation_mask[rr, cc]
#         unique_labels = np.unique(existing_labels[existing_labels > 0])
#         if len(unique_labels) == 1:
#             label_to_use = unique_labels[0]
#         else:
#             label_to_use = current_label
#             current_label += 1
#         segmentation_mask[rr, cc] = label_to_use
    
#     cleaned = remove_small_objects(segmentation_mask, min_size=min_area)
#     return cleaned, combined, contours

# Updated to include segmentation percentile
def segment_cells(image: np.ndarray, min_area: int = 10, use_percentile: bool = False, 
                 percentile: float = 87, crop: tuple = None) -> tuple:
    """Segment cells using contour-based approach.
    
    Args:
        image (np.ndarray): Preprocessed image.
        min_area (int): Minimum area for objects.
        use_percentile (bool): Use percentile thresholding instead of Otsu.
        percentile (float): Percentile for thresholding (if use_percentile=True).
        crop (tuple): Crop region (y1, y2, x1, x2) or None for full image.
    
    Returns:
        tuple: (Labeled segmentation mask, combined edge image, contours).
    """
    import logging
    logger = logging.getLogger(__name__)
    if crop:
        y1, y2, x1, x2 = crop
        image = image[y1:y2, x1:x2]
    
    image = invert(image)
    combined, weights = apply_edge_filters(image)
    
    if use_percentile:
        thresh = np.percentile(combined, percentile)
        logger.debug(f"Using percentile threshold: {thresh} (percentile={percentile})")
    else:
        thresh = threshold_otsu(combined)
        logger.debug(f"Using Otsu threshold: {thresh}")
    
    contours = find_contours(combined, thresh)
    logger.debug(f"Found {len(contours)} contours")
    
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
    
    cleaned = remove_small_objects(segmentation_mask, min_size=min_area)
    logger.debug(f"After filtering, {len(np.unique(cleaned)) - 1} objects remain")
    return cleaned, combined, contours

def segment_cells_cellpose(image: np.ndarray) -> np.ndarray:
    """Segment cells using Cellpose 4.0.6+ (optional).
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Labeled segmentation mask.
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose is not installed. Run 'pip install cellpose'.")
    
    image = image.astype(np.float32)
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = models.CellposeModel(gpu=(device.type == "mps"), device=device)
        masks, _, _, _ = model.eval([scharr(image)], diameter=None, channels=[0, 0])
    except Exception as e:
        print(f"MPS failed: {e}. Falling back to CPU.")
        device = torch.device个体
        model = models.CellposeModel(gpu=False, device=device)
        masks, _, _, _ = model.eval([scharr(image)], diameter=None, channels=[0, 0])
    
    return masks[0]

def refine_with_random_walker(image: np.ndarray, labeled: np.ndarray, beta: float = 1000) -> np.ndarray:
    """Refine segmentation using random walker.
    
    Args:
        image (np.ndarray): Input image (inverted).
        labeled (np.ndarray): Initial labeled mask.
        beta (float): Random walker stiffness parameter.
    
    Returns:
        np.ndarray: Refined probability map.
    """
    regions = regionprops(labeled)
    centres = [region.centroid for region in regions]
    centres_img = np.zeros_like(labeled)
    for i, (y, x) in enumerate(centres, 1):
        cy, cx = int(round(y)), int(round(x))
        if 0 <= cy < centres_img.shape[0] and 0 <= cx < centres_img.shape[1]:
            centres_img[cy, cx] = i
    
    confident_cell = centres_img > 0
    confident_bkg = labeled == 0
    markers = np.zeros_like(labeled)
    markers[confident_bkg] = 1
    markers[confident_cell] = centres_img[confident_cell]
    
    pb = random_walker(image, markers, beta=beta, mode='bf', return_full_prob=True)
    return pb[1]