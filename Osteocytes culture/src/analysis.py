import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize

def analyze_cells(labeled_mask: np.ndarray, image: np.ndarray) -> pd.DataFrame:
    """Analyze segmented cells.
    
    Args:
        labeled_mask (np.ndarray): Labeled cell mask.
        image (np.ndarray): Original or preprocessed image.
    
    Returns:
        pd.DataFrame: Cell metrics (area, intensity, etc.).
    """
    props = regionprops_table(
        labeled_mask, intensity_image=image,
        properties=['label', 'area', 'mean_intensity', 'eccentricity']
    )
    return pd.DataFrame(props)

def analyze_dendrites(binary_mask: np.ndarray, index: pd.Index = None) -> pd.DataFrame:
    """Analyze dendritic processes.
    
    Args:
        binary_mask (np.ndarray): Binary mask of cells.
        index (pd.Index, optional): Index to align with cell metrics DataFrame.
    
    Returns:
        pd.DataFrame: Dendritic metrics with matching index.
    """
    skeleton = skeletonize(binary_mask > 0)
    total_length = np.sum(skeleton)
    if index is None:
        return pd.DataFrame({'dendritic_length': [total_length]})
    else:
        # Create a DataFrame with the same index as cell_metrics
        return pd.DataFrame({'dendritic_length': [total_length] * len(index)}, index=index)