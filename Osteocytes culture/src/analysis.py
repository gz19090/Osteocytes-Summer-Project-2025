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
    
    Args:
        labeled_mask (np.ndarray): Labeled cell mask from segmentation.
        image (np.ndarray): Original or preprocessed image for intensity metrics.
    
    Returns:
        pd.DataFrame: Metrics including label, area, mean_intensity, eccentricity,
                      perimeter, major_axis_length, minor_axis_length, convex_area,
                      fractal_dimension, is_dendritic, bounding_box_area, equivalent_diameter,
                      extent, form_factor, solidity, and compactness for each cell.
    """
    # Define properties to compute
    properties = [
        'label', 'area', 'mean_intensity', 'eccentricity', 'perimeter',
        'major_axis_length', 'minor_axis_length', 'convex_area',
        'bbox_area', 'equivalent_diameter', 'extent'
    ]
    
    try:
        # Compute region properties
        props = regionprops_table(
            labeled_mask, intensity_image=image, properties=properties
        )
        # Convert to DataFrame
        df = pd.DataFrame(props)
        
        # Compute additional metrics
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
                    # Fractal dimension: D = 2 * log(perimeter) / log(area)
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
                    # Form factor: 4 * π * area / perimeter^2
                    ff = (4 * np.pi * area) / (perimeter ** 2)
                    form_factors.append(ff)
                    # Solidity: area / convex_area
                    solidity = area / convex_area if convex_area > 0 else np.nan
                    solidities.append(solidity)
                    # Compactness: perimeter^2 / (4 * π * area)
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    compactnesses.append(compactness)
                    # Dendritic flag: area/convex_area < 0.8
                    is_dendritic.append(solidity < 0.8)
            except Exception as e:
                logger.warning(f"Metrics failed for label {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
                form_factors.append(np.nan)
                solidities.append(np.nan)
                compactnesses.append(np.nan)
                is_dendritic.append(False)
        
        # Add computed metrics to DataFrame
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
    
    Args:
        labeled_mask (np.ndarray): Labeled cell mask (each cell has a unique label).
        index (pd.Index, optional): Index to align with cell metrics DataFrame.
    
    Returns:
        pd.DataFrame: Dendritic length for each cell, aligned with the provided index.
    """
    try:
        if index is None:
            # Compute total dendritic length for the entire mask
            skeleton = skeletonize(labeled_mask > 0)
            total_length = np.sum(skeleton)
            return pd.DataFrame({'dendritic_length': [total_length]})
        else:
            # Compute dendritic length per cell
            dendritic_lengths = []
            for label in index:
                try:
                    # Skeletonize individual cell mask
                    cell_mask = labeled_mask == label
                    skeleton = skeletonize(cell_mask)
                    length = np.sum(skeleton)
                    dendritic_lengths.append(length)
                except Exception as e:
                    logger.warning(f"Dendritic length failed for label {label}: {e}. Using NaN.")
                    dendritic_lengths.append(np.nan)
            return pd.DataFrame({'dendritic_length': dendritic_lengths}, index=index)
    except Exception as e:
        logger.error(f"Error in analyze_dendrites: {e}")
        return pd.DataFrame({'dendritic_length': [np.nan] * (len(index) if index is not None else 1)})