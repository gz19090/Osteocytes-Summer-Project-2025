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
                      and fractal_dimension for each cell.
    """
    # Define properties to compute
    properties = [
        'label', 'area', 'mean_intensity', 'eccentricity',
        'perimeter', 'major_axis_length', 'minor_axis_length', 'convex_area'
    ]
    
    try:
        # Compute region properties
        props = regionprops_table(
            labeled_mask, intensity_image=image, properties=properties
        )
        # Convert to DataFrame
        df = pd.DataFrame(props)
        
        # Compute fractal dimension using perimeter-area relationship
        fractal_dimensions = []
        for _, row in df.iterrows():
            try:
                area = row['area']
                perimeter = row['perimeter']
                if area <= 0 or perimeter <= 0:
                    logger.warning(f"Invalid area ({area}) or perimeter ({perimeter}) for label {row['label']}. Using NaN.")
                    fractal_dimensions.append(np.nan)
                else:
                    # Approximate fractal dimension: D = 2 * log(perimeter) / log(area)
                    fd = 2 * np.log(perimeter) / np.log(area)
                    fractal_dimensions.append(fd)
            except Exception as e:
                logger.warning(f"Fractal dimension failed for label {row['label']}: {e}. Using NaN.")
                fractal_dimensions.append(np.nan)
        
        # Add fractal dimension to DataFrame
        df['fractal_dimension'] = fractal_dimensions
        
        return df
    except Exception as e:
        logger.error(f"Error in analyze_cells: {e}")
        return pd.DataFrame()

def analyze_dendrites(binary_mask: np.ndarray, index: pd.Index = None) -> pd.DataFrame:
    """Analyze dendritic processes by computing total skeleton length.
    
    Args:
        binary_mask (np.ndarray): Binary mask of cells (e.g., labeled_mask > 0).
        index (pd.Index, optional): Index to align with cell metrics DataFrame.
    
    Returns:
        pd.DataFrame: Dendritic length aligned with the provided index.
    """
    try:
        skeleton = skeletonize(binary_mask > 0)
        total_length = np.sum(skeleton)
        if index is None:
            return pd.DataFrame({'dendritic_length': [total_length]})
        else:
            return pd.DataFrame({'dendritic_length': [total_length] * len(index)}, index=index)
    except Exception as e:
        logger.error(f"Error in analyze_dendrites: {e}")
        return pd.DataFrame({'dendritic_length': [np.nan] * (len(index) if index is not None else 1)})