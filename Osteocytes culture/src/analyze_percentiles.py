# analyze_percentiles.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.signal import savgol_filter
from skimage.measure import regionprops

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust the path to the root of the project
project_root = Path('/Users/diana/Desktop/Osteocytes-Summer-Project-2025/Osteocytes culture')
sys.path.append(str(project_root))
logger.info(f"Script path: {Path(__file__).resolve()}")
logger.info(f"Project root: {project_root}")
logger.info(f"sys.path: {sys.path}")

# Verify src directory exists
src_dir = project_root / 'src'
if not src_dir.exists():
    logger.error(f"src directory not found at {src_dir}. Please verify project structure.")
    raise FileNotFoundError(f"src directory not found at {src_dir}")

from src.image_utils import load_video, correct_background, apply_fourier_filter
from src.segmentation import apply_edge_filters, segment_cells
from src.visualization import plot_segmentation

def analyze_percentiles(video_path: str, percentiles: range, max_frames: int = 1, min_area: int = 50):
    """Analyze the number of cells detected for different percentiles and collect labeled and original images for percentiles 90-95.
    Args:
        video_path (str): Path to the input video.
        percentiles (range): Range of percentiles to test (e.g., range(80, 100)).
        max_frames (int): Number of frames to process (default: 1).
        min_area (int): Minimum area for segmented objects (default: 50).
    Returns:
        tuple: (pd.DataFrame with 'percentile' and 'num_cells', dict of labeled images, np.ndarray of original image).
    """
    results = []
    labeled_images = {}  # Store labeled images for percentiles 90-95
    original_image = None  # Store the filtered image for the first frame
    output_dir = project_root / 'results/figures/percentile_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Loading video: {video_path}")
        frames = load_video(video_path)
        frames_to_process = frames[:max_frames]
        logger.info(f"Processing {len(frames_to_process)} frames from {video_path}")
        
        for percentile in percentiles:
            logger.info(f"Testing percentile: {percentile}")
            num_cells_per_frame = []
            for frame_idx, frame in enumerate(frames_to_process):
                try:
                    corrected = correct_background(frame)
                    filtered = apply_fourier_filter(corrected)
                    combined, _ = apply_edge_filters(filtered)
                    labeled, _, _ = segment_cells(
                        filtered, min_area=min_area, use_percentile=True, percentile=percentile, crop=None)
                    num_cells = len(np.unique(labeled)) - 1  # Subtract background (label 0)
                    num_cells_per_frame.append(num_cells)
                    logger.debug(f"Frame {frame_idx}: {num_cells} cells detected at percentile {percentile}")
                    
                    # Store labeled and original image for percentiles 90-95 (first frame only)
                    if percentile >= 90 and percentile <= 95 and frame_idx == 0:
                        labeled_images[percentile] = labeled
                        if original_image is None:
                            original_image = filtered
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx} at percentile {percentile}: {e}")
                    num_cells_per_frame.append(0)
            avg_num_cells = np.mean(num_cells_per_frame) if num_cells_per_frame else 0
            results.append({'percentile': percentile, 'num_cells': avg_num_cells})
            logger.info(f"Average number of cells for percentile {percentile}: {avg_num_cells:.2f}")
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return pd.DataFrame(), {}, None
    
    return pd.DataFrame(results), labeled_images, original_image

def find_optimal_percentile(results: pd.DataFrame) -> float:
    """Find the optimal percentile based on the elbow point around 90 with a drop-off.
    Args:
        results (pd.DataFrame): DataFrame with 'percentile' and 'num_cells' columns.
    Returns:
        float: Optimal percentile value.
    """
    if results.empty:
        logger.warning("No results to analyze for optimal percentile. Defaulting to 90.")
        return 90.0
    # Sort by percentile to ensure order
    results = results.sort_values('percentile')
    num_cells = results['num_cells'].values
    percentiles = results['percentile'].values
    
    # Smooth the cell counts to reduce noise and detect trends
    if len(num_cells) >= 3:
        num_cells_smoothed = savgol_filter(num_cells, window_length=3, polyorder=1)
    else:
        num_cells_smoothed = num_cells
    
    # Compute the first differences (deltas) to detect changes
    deltas = np.diff(num_cells_smoothed)
    
    # Find the elbow around 90: look for the percentile near 90 where there's a drop-off (negative delta or flattening)
    elbow_candidate = None
    for i in range(len(percentiles) - 1):
        if percentiles[i] >= 90 and deltas[i] < 0:  # Drop-off after 90
            elbow_candidate = percentiles[i]
            break
    if elbow_candidate is None:
        # If no drop-off, take the percentile near 90 with the highest cell count
        idx_near_90 = np.argmin(np.abs(percentiles - 90))
        elbow_candidate = percentiles[idx_near_90]
    logger.info(f"Optimal percentile: {elbow_candidate} (num_cells: {num_cells[percentiles.tolist().index(elbow_candidate)]:.2f}, elbow with drop-off)")
    return float(elbow_candidate)

def plot_percentile_results(results: pd.DataFrame, output_path: str, dpi: int = 150):
    """Plot the number of cells versus percentile with the optimal percentile highlighted.
    Args:
        results (pd.DataFrame): DataFrame with 'percentile' and 'num_cells' columns.
        output_path (str): Path to save the plot.
        dpi (int): Resolution of the output image (default: 150).
    """
    if results.empty:
        logger.warning("No results to plot.")
        return
    optimal_percentile = find_optimal_percentile(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['percentile'], results['num_cells'], marker='o', label='Number of Cells')
    plt.axvline(x=optimal_percentile, color='red', linestyle='--', label=f'Optimal Percentile: {optimal_percentile}')
    plt.xlabel('Percentile')
    plt.ylabel('Average Number of Cells')
    plt.title('Number of Cells Detected vs. Percentile')
    plt.grid(True)
    plt.legend()
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved percentile plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save percentile plot to {output_path}: {e}")
    plt.close()

def plot_labeled_comparison(labeled_images: dict, original_image: np.ndarray, output_path: str, dpi: int = 150, fontsize: int = 6):
    """Plot labeled cell images and original image for percentiles 90-95 side by side in a single figure.
    Args:
        labeled_images (dict): Dictionary mapping percentiles (90-95) to labeled images.
        original_image (np.ndarray): Original (filtered) image to display.
        output_path (str): Path to save the comparison figure.
        dpi (int): Resolution of the output image (default: 150).
        fontsize (int): Font size for cell ID annotations (default: 6).
    """
    if not labeled_images or original_image is None:
        logger.warning("No labeled images or original image provided for comparison plot.")
        return
    
    plt.figure(figsize=(18, 6))
    for i, percentile in enumerate(range(90, 96)):
        # Top row: Labeled images
        plt.subplot(2, 6, i + 1)
        if percentile not in labeled_images:
            logger.warning(f"No labeled image for percentile {percentile}. Skipping subplot.")
            plt.axis('off')
            continue
        labeled = labeled_images[percentile]
        num_cells = len(np.unique(labeled)) - 1
        plt.imshow(labeled, cmap='nipy_spectral')
        regions = regionprops(labeled)
        for region in regions:
            y, x = region.centroid
            plt.text(x, y, str(region.label), color='white', fontsize=fontsize, ha='center', va='center')
        plt.title(f'Percentile {percentile}\n({num_cells} cells)')
        plt.axis('off')
        
        # Bottom row: Original image (repeated)
        plt.subplot(2, 6, i + 7)  # 6 columns, so +7 for second row
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
    
    plt.tight_layout()
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved labeled comparison plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save labeled comparison plot to {output_path}: {e}")
    plt.close()

def main():
    """Main function to analyze percentiles, plot results, and create comparison figure for percentiles 90-95."""
    video_path = str(project_root / 'data/raw/mutant/Confluence_Single movie_30.03.2025_no mask_G5_1.mp4')
    output_dir = project_root / 'results/figures/percentile_analysis'
    output_path = str(output_dir / 'percentile_vs_num_cells.png')
    comparison_path = str(output_dir / 'percentile_comparison_90_95.png')
    
    # Check if video exists
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Define percentile range
    percentiles = range(80, 100)
    logger.info(f"Analyzing percentiles: {list(percentiles)}")
    
    # Analyze percentiles and collect labeled and original images
    results, labeled_images, original_image = analyze_percentiles(video_path, percentiles, max_frames=1, min_area=50)
    
    # Save results to CSV
    if not results.empty:
        try:
            csv_path = output_dir / 'percentile_results.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(csv_path, index=False)
            logger.info(f"Saved percentile results to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save percentile results to {csv_path}: {e}")
    
    # Plot cell count results with optimal percentile highlighted
    plot_percentile_results(results, output_path)
    
    # Plot side-by-side labeled and original images for percentiles 90-95
    plot_labeled_comparison(labeled_images, original_image, comparison_path)

if __name__ == '__main__':
    main()