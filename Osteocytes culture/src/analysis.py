# analysis.py

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects, skeletonize, opening, disk
import logging
from pathlib import Path
import imageio.v2 as imageio
from scipy import ndimage as ndi

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
                else:
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
                    ff = (4 * np.pi * area) / (perimeter ** 2)
                    form_factors.append(ff)
                    solidity = area / convex_area if convex_area > 0 else np.nan
                    solidities.append(solidity)
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    compactnesses.append(compactness)
            except Exception as e:
                logger.warning(f"Metrics failed for label {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
                form_factors.append(np.nan)
                solidities.append(np.nan)
                compactnesses.append(np.nan)
        df['fractal_dimension'] = fractal_dimensions
        df['form_factor'] = form_factors
        df['solidity'] = solidities
        df['compactness'] = compactnesses
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

def _protrusion_skeleton(cell_mask: np.ndarray,
                         soma_frac: float = 0.18,
                         min_component_size: int = 8) -> np.ndarray:
    """
    Build a dendrite-only skeleton by removing the thick soma via morphological opening.

    Steps:
      1) Estimate an effective soma radius from cell area: r ≈ soma_frac * equiv_diameter.
      2) opened = opening(cell, disk(r))  -> soma core (thin connections broken).
      3) protrusions = cell & ~opened     -> thin processes (dendrites).
      4) Remove tiny specks and skeletonize.
    """
    cell = cell_mask.astype(bool, copy=False)
    if cell.sum() == 0:
        return np.zeros_like(cell, dtype=bool)

    area = int(cell.sum())
    # equiv_diameter = diameter of a circle with same area
    equiv_d = 2.0 * np.sqrt(area / np.pi)
    r = max(2, int(round(soma_frac * equiv_d)))  # adaptive, ≥2 px

    core = opening(cell, disk(r))
    protrusions = cell & ~core

    # clean tiny islands in protrusions before skeletonizing
    protrusions = remove_small_objects(protrusions, min_size=min_component_size)

    skel = skeletonize(protrusions)
    return skel

def _count_dendrites_from_protrusions(skel: np.ndarray,
                                      min_len_px: int = 5) -> int:
    """
    Count connected skeleton components as dendrites, requiring a minimum pixel length.
    """
    skel = skel.astype(bool, copy=False)
    if skel.sum() < min_len_px:
        return 0

    # Label connected skeleton components
    lbl, ncomp = ndi.label(skel, structure=np.ones((3, 3), dtype=int))
    if ncomp == 0:
        return 0

    count = 0
    for cid in range(1, ncomp + 1):
        comp = (lbl == cid)
        if comp.sum() >= min_len_px:
            count += 1
    return int(count)


def analyze_dendrite_count(labeled_mask: np.ndarray,
                           index: pd.Index = None,
                           output_dir: str = None,
                           soma_frac: float = 0.18,
                           min_component_size: int = 8,
                           min_len_px: int = 5) -> pd.DataFrame:
    """
    Compute dendrite COUNT per cell via morphological-opening-based protrusions.

    - soma_frac: fraction of equivalent diameter to set opening radius (typ. 0.15–0.25).
    - min_component_size: remove tiny protrusion blobs before skeletonization.
    - min_len_px: minimal skeleton pixels per protrusion to count as a dendrite.
    """
    try:
        if index is None:
            # Whole-mask total (rarely needed); treat all foreground as one cell
            cell = (labeled_mask > 0)
            skel = _protrusion_skeleton(cell, soma_frac=soma_frac, min_component_size=min_component_size)
            total = _count_dendrites_from_protrusions(skel, min_len_px=min_len_px)
            return pd.DataFrame({'dendrite_count': [total]})

        counts = []
        for lbl in index:
            try:
                cell = (labeled_mask == int(lbl))
                skel = _protrusion_skeleton(cell, soma_frac=soma_frac, min_component_size=min_component_size)
                cnt = _count_dendrites_from_protrusions(skel, min_len_px=min_len_px)
                counts.append(cnt)
            except Exception as e:
                logger.warning(f"Dendrite count failed for label {lbl}: {e}. Using 0.")
                counts.append(0)
        return pd.DataFrame({'dendrite_count': counts}, index=index)
    except Exception as e:
        logger.error(f"Error in analyze_dendrite_count: {e}")
        return pd.DataFrame({'dendrite_count': [0] * (len(index) if index is not None else 1)})