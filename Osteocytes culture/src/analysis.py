# analysis.py

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize
import logging

# Set up logging for debugging
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
    # Define properties to compute using regionprops_table
    properties = [
        'label', 'area', 'mean_intensity', 'eccentricity', 'perimeter',
        'major_axis_length', 'minor_axis_length', 'convex_area',
        'bbox_area', 'equivalent_diameter', 'extent'
    ]
    
    try:
        # Compute region properties from the labeled mask and intensity image
        props = regionprops_table(
            labeled_mask, intensity_image=image, properties=properties
        )
        # Convert the properties dictionary to a Pandas DataFrame
        df = pd.DataFrame(props)
        
        # Initialize lists for additional computed metrics
        fractal_dimensions = []
        is_dendritic = []
        form_factors = []
        solidities = []
        compactnesses = []
        
        # Iterate over each row (cell) to compute additional metrics
        for _, row in df.iterrows():
            try:
                area = row['area']
                perimeter = row['perimeter']
                convex_area = row['convex_area']
                
                # Check for invalid values that could cause log(0) or division by zero
                if area <= 0 or perimeter <= 0:
                    logger.warning(f"Invalid area ({area}) or perimeter ({perimeter}) for label {row['label']}. Using NaN.")
                    fractal_dimensions.append(np.nan)
                    form_factors.append(np.nan)
                    solidities.append(np.nan)
                    compactnesses.append(np.nan)
                    is_dendritic.append(False)
                else:
                    # Fractal dimension: Measures shape complexity (D = 2 * log(perimeter) / log(area))
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
                    
                    # Form factor: Measures circularity (closer to 1 for circles, lower for irregular shapes)
                    ff = (4 * np.pi * area) / (perimeter ** 2)
                    form_factors.append(ff)
                    
                    # Solidity: Proportion of cell area to its convex hull (lower for shapes with protrusions)
                    solidity = area / convex_area if convex_area > 0 else np.nan
                    solidities.append(solidity)
                    
                    # Compactness: Measures how tightly packed the shape is (higher for irregular shapes)
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    compactnesses.append(compactness)
                    
                    # Dendritic flag: True if solidity < 0.8, indicating dendritic morphology with protrusions
                    is_dendritic.append(solidity < 0.8)
            except Exception as e:
                logger.warning(f"Metrics failed for label {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
                form_factors.append(np.nan)
                solidities.append(np.nan)
                compactnesses.append(np.nan)
                is_dendritic.append(False)
        
        # Add computed metrics as new columns to the DataFrame
        df['fractal_dimension'] = fractal_dimensions
        df['form_factor'] = form_factors
        df['solidity'] = solidities
        df['compactness'] = compactnesses
        df['is_dendritic'] = is_dendritic
        
        return df
    except Exception as e:
        logger.error(f"Error in analyze_cells: {e}")
        return pd.DataFrame()

def analyze_dendrites(labeled_mask: np.ndarray, index: pd.Index = None) -> pd.DataFrame:
    """Analyze dendritic processes by computing skeleton length per cell.
    
    This function skeletonizes the binary mask of each cell (or the entire mask if no index is provided)
    to estimate dendritic process length as the sum of skeleton pixels. It uses per-cell masks to avoid overestimation
    and handles errors by setting length to 0.
    
    Args:
        labeled_mask (np.ndarray): Labeled cell mask (each cell has a unique label).
        index (pd.Index, optional): Index to align with cell metrics DataFrame (for per-cell analysis).
    
    Returns:
        pd.DataFrame: Dendritic length for each cell, aligned with the provided index.
    """
    try:
        if index is None:
            # Compute total dendritic length for the entire mask (binary threshold at >0)
            skeleton = skeletonize(labeled_mask > 0)
            total_length = np.sum(skeleton)
            return pd.DataFrame({'dendritic_length': [total_length]})
        else:
            # Compute dendritic length per cell
            dendritic_lengths = []
            for label in index:
                try:
                    # Create binary mask for the specific cell label and ensure uint8 type for skeletonize
                    cell_mask = (labeled_mask == label).astype(np.uint8)
                    # Skeletonize to reduce to 1-pixel wide representation
                    skeleton = skeletonize(cell_mask)
                    # Sum the skeleton pixels to get length
                    length = np.sum(skeleton)
                    dendritic_lengths.append(length)
                except Exception as e:
                    logger.warning(f"Dendritic length failed for label {label}: {e}. Using 0.")
                    dendritic_lengths.append(0)
            return pd.DataFrame({'dendritic_length': dendritic_lengths}, index=index)
    except Exception as e:
        logger.error(f"Error in analyze_dendrites: {e}")
        return pd.DataFrame({'dendritic_length': [0] * (len(index) if index is not None else 1)})