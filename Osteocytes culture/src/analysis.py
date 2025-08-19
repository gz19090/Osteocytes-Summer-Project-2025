# analysis.py

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize
import logging
from pathlib import Path
import imageio.v2 as imageio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_cells(labeled_mask: np.ndarray, image: np.ndarray) -> pd.DataFrame:
    """Analyze segmented cells to compute morphological metrics.
    This function uses skimage's regionprops_table to extract basic properties from the labeled mask
    and intensity image, then computes additional shape metrics like fractal dimension, form factor,
    solidity, compactness, and a dendritic flag. It handles invalid regions (e.g., zero area/perimeter)
    by assigning NaN or False values.
    Args:
        labeled_mask (np.ndarray): Labeled cell mask from segmentation (2D array where each unique value >0 represents a cell).
        image (np.ndarray): Original or preprocessed image for intensity metrics (must match labeled_mask shape).
    Returns:
        pd.DataFrame: Metrics including label, area, mean_intensity, eccentricity,
                      perimeter, major_axis_length, minor_axis_length, convex_area,
                      fractal_dimension, is_dendritic, bounding_box_area (as bbox_area),
                      equivalent_diameter, extent, form_factor, solidity, and compactness for each cell.
    """
    properties = [
        'label', 'area', 'mean_intensity', 'eccentricity', 'perimeter',
        'major_axis_length', 'minor_axis_length', 'convex_area',
        'bbox_area', 'equivalent_diameter', 'extent'
    ]
    try:
        props = regionprops_table(
            labeled_mask, intensity_image=image, properties=properties
        )
        df = pd.DataFrame(props)
        fractal_dimensions = []
        is_dendritic = []
        form_factors = []
        solidities = []
        compactnesses = []
        for _, row in df.iterrows():
            try:
                area = row['area']
                perimeter = row['perimeter']
                convex_area = row['convex_area']
                if area <= 0 or perimeter <= 0:
                    logger.warning(f"Invalid area ({area}) or perimeter ({perimeter}) for label {row['label']}. Using NaN.")
                    fractal_dimensions.append(np.nan)
                    form_factors.append(np.nan)
                    solidities.append(np.nan)
                    compactnesses.append(np.nan)
                    is_dendritic.append(False)
                else:
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
                    ff = (4 * np.pi * area) / (perimeter ** 2)
                    form_factors.append(ff)
                    solidity = area / convex_area if convex_area > 0 else np.nan
                    solidities.append(solidity)
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    compactnesses.append(compactness)
                    is_dendritic.append(solidity < 0.8)
            except Exception as e:
                logger.warning(f"Metrics failed for label {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
                form_factors.append(np.nan)
                solidities.append(np.nan)
                compactnesses.append(np.nan)
                is_dendritic.append(False)
        df['fractal_dimension'] = fractal_dimensions
        df['form_factor'] = form_factors
        df['solidity'] = solidities
        df['compactness'] = compactnesses
        df['is_dendritic'] = is_dendritic
        return df
    except Exception as e:
        logger.error(f"Error in analyze_cells: {e}")
        return pd.DataFrame()

def skeletonise(binary_mask: np.ndarray, output_dir: str = None) -> np.ndarray:
    """Perform skeletonization on a binary mask and optionally save the result.
    Args:
        binary_mask (np.ndarray): Binary mask (2D, uint8 or bool) of a single cell or region.
        output_dir (str, optional): Directory to save the skeleton image. If None, no image is saved.
    Returns:
        np.ndarray: Skeletonized image (2D, bool).
    """
    try:
        if not isinstance(binary_mask, np.ndarray) or binary_mask.ndim != 2:
            logger.error(f"Invalid binary mask shape: {binary_mask.shape}")
            raise ValueError("Binary mask must be a 2D NumPy array.")
        # Ensure binary mask is uint8 or bool
        binary_mask = binary_mask.astype(np.uint8) if binary_mask.dtype != bool else binary_mask
        skeleton = skeletonize(binary_mask)
        if output_dir:
            output_path = Path(output_dir) / "skeleton.tif"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(str(output_path), skeleton.astype(np.uint8) * 255, format='TIFF')
            logger.info(f"Saved skeleton to {output_path}")
        logger.debug(f"Skeletonized image shape: {skeleton.shape}, non-zero pixels: {np.sum(skeleton)}")
        return skeleton
    except Exception as e:
        logger.error(f"Error in skeletonise: {e}")
        raise

def analyze_dendrites(labeled_mask: np.ndarray, index: pd.Index = None, output_dir: str = None) -> pd.DataFrame:
    """Analyze dendritic processes by computing skeleton length per cell.
    This function skeletonizes the binary mask of each cell (or the entire mask if no index is provided)
    to estimate dendritic process length as the sum of skeleton pixels. It uses per-cell masks to avoid overestimation
    and handles errors by setting length to 0.
    Args:
        labeled_mask (np.ndarray): Labeled cell mask (each cell has a unique label).
        index (pd.Index, optional): Index to align with cell metrics DataFrame (for per-cell analysis).
        output_dir (str, optional): Directory to save skeleton images (one per cell if index is provided).
    Returns:
        pd.DataFrame: Dendritic length for each cell, aligned with the provided index.
    """
    try:
        if index is None:
            skeleton = skeletonise(labeled_mask > 0, output_dir)
            total_length = np.sum(skeleton)
            return pd.DataFrame({'dendritic_length': [total_length]})
        else:
            dendritic_lengths = []
            for label in index:
                try:
                    cell_mask = (labeled_mask == label).astype(np.uint8)
                    skeleton_dir = f"{output_dir}/label_{label}" if output_dir else None
                    skeleton = skeletonise(cell_mask, skeleton_dir)
                    length = np.sum(skeleton)
                    dendritic_lengths.append(length)
                except Exception as e:
                    logger.warning(f"Dendritic length failed for label {label}: {e}. Using 0.")
                    dendritic_lengths.append(0)
            return pd.DataFrame({'dendritic_length': dendritic_lengths}, index=index)
    except Exception as e:
        logger.error(f"Error in analyze_dendrites: {e}")
        return pd.DataFrame({'dendritic_length': [0] * (len(index) if index is not None else 1)})