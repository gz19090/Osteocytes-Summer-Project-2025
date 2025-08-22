# analysis.py

"""
analysis.py
----------------
This file contains functions for analyzing osteocyte cell images in the osteocyte culture project.
It processes segmented cell masks to compute morphological metrics (e.g., area, perimeter, solidity)
and counts dendrites by skeletonizing cell protrusions. The functions are designed to work with 2D
grayscale images and labeled masks, producing metrics for statistical analysis and visualization.
Key functions include:
- analyze_cells: Computes shape and intensity metrics for segmented cells.
- skeletonise: Creates a thin skeleton of a cell's binary mask for structural analysis.
- analyze_dendrite_count: Counts dendrites by isolating and skeletonizing cell protrusions.
Used in the main_workflow.py to generate metrics for wildtype and mutant osteocyte videos.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects, skeletonize, opening, disk
import logging
from pathlib import Path
import imageio.v2 as imageio
from scipy import ndimage as ndi

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_cells(labeled_mask: np.ndarray, image: np.ndarray) -> pd.DataFrame:
    """Analyze cells in a labeled image to calculate shape and intensity metrics.
    This function takes a labeled mask (where each cell has a unique number) and an image,
    then computes metrics like area, intensity, and shape properties for each cell.
    It also calculates extra metrics like fractal dimension and solidity to describe cell complexity.
    
    Args:
        labeled_mask (np.ndarray): 2D array where each cell is marked with a unique number > 0.
        image (np.ndarray): 2D grayscale image for intensity measurements (same size as labeled_mask).
    
    Returns:
        pd.DataFrame: Table with metrics for each cell, including area, intensity, shape, and more.
    """
    # List of basic cell properties to measure
    properties = [
        'label', 'area', 'mean_intensity', 'eccentricity', 'perimeter',
        'major_axis_length', 'minor_axis_length', 'convex_area',
        'bbox_area', 'equivalent_diameter', 'extent'
    ]
    
    try:
        # Measure basic properties for all cells in the labeled mask
        props = regionprops_table(labeled_mask, intensity_image=image, properties=properties)
        df = pd.DataFrame(props)  # Convert measurements to a table
        
        # Lists to store additional shape metrics
        fractal_dimensions = []
        form_factors = []
        solidities = []
        compactnesses = []
        
        # Calculate extra metrics for each cell
        for _, row in df.iterrows():
            try:
                area = row['area']
                perimeter = row['perimeter']
                convex_area = row['convex_area']
                
                # Check if area or perimeter is invalid (zero or negative)
                if area <= 0 or perimeter <= 0:
                    logger.warning(f"Invalid area ({area}) or perimeter ({perimeter}) for cell {row['label']}. Using NaN.")
                    fractal_dimensions.append(np.nan)
                    form_factors.append(np.nan)
                    solidities.append(np.nan)
                    compactnesses.append(np.nan)
                else:
                    # Calculate fractal dimension (measures edge complexity)
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
                    
                    # Calculate form factor (how circular the cell is)
                    ff = (4 * np.pi * area) / (perimeter ** 2)
                    form_factors.append(ff)
                    
                    # Calculate solidity (how filled the cell is compared to its convex shape)
                    solidity = area / convex_area if convex_area > 0 else np.nan
                    solidities.append(solidity)
                    
                    # Calculate compactness (how compact the cell shape is)
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    compactnesses.append(compactness)
            
            except Exception as e:
                # Log error and use NaN for failed calculations
                logger.warning(f"Metrics failed for cell {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
                form_factors.append(np.nan)
                solidities.append(np.nan)
                compactnesses.append(np.nan)
        
        # Add calculated metrics to the table
        df['fractal_dimension'] = fractal_dimensions
        df['form_factor'] = form_factors
        df['solidity'] = solidities
        df['compactness'] = compactnesses
        return df
    
    except Exception as e:
        # Log error and return empty table if analysis fails
        logger.error(f"Error in analyze_cells: {e}")
        return pd.DataFrame()

def skeletonise(binary_mask: np.ndarray, output_dir: str = None) -> np.ndarray:
    """Create a skeleton (thin outline) of a cell from its binary mask.
    This function takes a binary mask (1s for the cell, 0s for background) and
    reduces it to a 1-pixel-wide skeleton to highlight the cell's structure.
    Optionally saves the skeleton as an image.
    
    Args:
        binary_mask (np.ndarray): 2D binary mask (uint8 or bool) of a single cell.
        output_dir (str, optional): Folder to save the skeleton image. If None, no saving.
    
    Returns:
        np.ndarray: Skeletonized image (2D, bool).
    """
    try:
        # Check if the input mask is a valid 2D array
        if not isinstance(binary_mask, np.ndarray) or binary_mask.ndim != 2:
            logger.error(f"Invalid binary mask shape: {binary_mask.shape}")
            raise ValueError("Binary mask must be a 2D NumPy array.")
        
        # Convert mask to uint8 if it's not boolean
        binary_mask = binary_mask.astype(np.uint8) if binary_mask.dtype != bool else binary_mask
        
        # Create skeleton (thin lines representing the cell's structure)
        skeleton = skeletonize(binary_mask)
        
        # Save skeleton image if output_dir is provided
        if output_dir:
            output_path = Path(output_dir) / "skeleton.tif"
            Path(output_dir).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
            # Convert skeleton to uint8 (0 or 255) for clear image saving
            imageio.imwrite(str(output_path), skeleton.astype(np.uint8) * 255, format='TIFF')
            logger.info(f"Saved skeleton to {output_path}")
        
        # Log skeleton details for debugging
        logger.debug(f"Skeletonized image shape: {skeleton.shape}, non-zero pixels: {np.sum(skeleton)}")
        return skeleton
    
    except Exception as e:
        # Log error and raise it to stop execution
        logger.error(f"Error in skeletonise: {e}")
        raise

def _protrusion_skeleton(cell_mask: np.ndarray, soma_frac: float = 0.18, min_component_size: int = 8) -> np.ndarray:
    """Create a skeleton of a cell's dendrites by removing the main cell body.
    This function isolates thin processes (dendrites) by:
    1) Estimating the cell body size based on the cell's area.
    2) Removing the cell body using morphological opening.
    3) Skeletonizing the remaining thin processes.
    
    Args:
        cell_mask (np.ndarray): Binary mask of a single cell (bool).
        soma_frac (float): Fraction of cell diameter to estimate cell body size (default: 0.18).
        min_component_size (int): Minimum size of protrusions to keep (default: 8 pixels).
    
    Returns:
        np.ndarray: Skeleton of the cell's dendrites (2D, bool).
    """
    # Convert mask to boolean
    cell = cell_mask.astype(bool, copy=False)
    
    # Return empty skeleton if the mask is empty
    if cell.sum() == 0:
        return np.zeros_like(cell, dtype=bool)
    
    # Calculate cell area
    area = int(cell.sum())
    
    # Estimate cell body radius from area (equivalent diameter = circle with same area)
    equiv_d = 2.0 * np.sqrt(area / np.pi)
    r = max(2, int(round(soma_frac * equiv_d)))  # Ensure radius is at least 2 pixels
    
    # Remove cell body (core) using morphological opening
    core = opening(cell, disk(r))
    
    # Isolate protrusions (dendrites) by subtracting the core
    protrusions = cell & ~core
    
    # Remove tiny noise in protrusions
    protrusions = remove_small_objects(protrusions, min_size=min_component_size)
    
    # Skeletonize the protrusions
    skel = skeletonize(protrusions)
    return skel

def _count_dendrites_from_protrusions(skel: np.ndarray, min_len_px: int = 5) -> int:
    """Count the number of dendrites in a skeleton by finding connected components.
    Each connected component (branch) with enough pixels is counted as a dendrite.
    
    Args:
        skel (np.ndarray): Skeletonized image of dendrites (bool).
        min_len_px (int): Minimum number of pixels for a branch to count as a dendrite (default: 5).
    
    Returns:
        int: Number of dendrites.
    """
    # Convert skeleton to boolean
    skel = skel.astype(bool, copy=False)
    
    # Return 0 if skeleton is too small
    if skel.sum() < min_len_px:
        return 0
    
    # Label connected components in the skeleton
    lbl, ncomp = ndi.label(skel, structure=np.ones((3, 3), dtype=int))
    
    # Return 0 if no components are found
    if ncomp == 0:
        return 0
    
    # Count components with enough pixels
    count = 0
    for cid in range(1, ncomp + 1):
        comp = (lbl == cid)
        if comp.sum() >= min_len_px:
            count += 1
    
    return int(count)

def analyze_dendrite_count(labeled_mask: np.ndarray, index: pd.Index = None, output_dir: str = None,
                          soma_frac: float = 0.18, min_component_size: int = 8, min_len_px: int = 5) -> pd.DataFrame:
    """Count dendrites for each cell in the labeled mask.
    This function processes each cell to isolate and skeletonize its dendrites,
    then counts the number of connected dendritic branches.
    
    Args:
        labeled_mask (np.ndarray): Labeled cell mask (each cell has a unique number).
        index (pd.Index, optional): List of cell labels to process.
        output_dir (str, optional): Folder to save skeleton images.
        soma_frac (float): Fraction of cell diameter for cell body estimation (default: 0.18).
        min_component_size (int): Minimum size of protrusions to keep (default: 8 pixels).
        min_len_px (int): Minimum pixels for a dendrite branch (default: 5).
    
    Returns:
        pd.DataFrame: Table with dendrite counts for each cell.
    """
    try:
        if index is None:
            # Treat entire mask as one cell (rare case)
            cell = (labeled_mask > 0)
            skel = _protrusion_skeleton(cell, soma_frac=soma_frac, min_component_size=min_component_size)
            total = _count_dendrites_from_protrusions(skel, min_len_px=min_len_px)
            return pd.DataFrame({'dendrite_count': [total]})
        
        # Process each cell in the index
        counts = []
        for lbl in index:
            try:
                # Extract mask for the current cell
                cell = (labeled_mask == int(lbl))
                # Create dendrite skeleton
                skel = _protrusion_skeleton(cell, soma_frac=soma_frac, min_component_size=min_component_size)
                # Count dendrites
                cnt = _count_dendrites_from_protrusions(skel, min_len_px=min_len_px)
                counts.append(cnt)
            except Exception as e:
                # Log error and use 0 if dendrite counting fails
                logger.warning(f"Dendrite count failed for cell {lbl}: {e}. Using 0.")
                counts.append(0)
        
        # Return table with dendrite counts
        return pd.DataFrame({'dendrite_count': counts}, index=index)
    
    except Exception as e:
        # Log error and return zeros if the entire function fails
        logger.error(f"Error in analyze_dendrite_count: {e}")
        return pd.DataFrame({'dendrite_count': [0] * (len(index) if index is not None else 1)})