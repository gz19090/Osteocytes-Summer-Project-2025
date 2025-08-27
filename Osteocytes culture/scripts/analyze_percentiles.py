# analyze_percentiles.py

# Import necessary libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.signal import savgol_filter
from skimage.measure import regionprops

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Adjust the path to the root of the project
project_root = Path('/Users/diana/Desktop/Osteocytes-Summer-Project-2025/Osteocytes culture')
sys.path.append(str(project_root))
logger.info(f"Project root: {project_root}")

# Verify src directory exists
src_dir = project_root / 'src'
if not src_dir.exists():
    logger.error(f"src directory not found at {src_dir}. Please verify project structure.")
    raise FileNotFoundError(f"src directory not found at {src_dir}")

from src.image_utils import load_video, correct_background, apply_fourier_filter
from src.segmentation import apply_edge_filters, segment_cells
from src.visualization import plot_segmentation

def analyze_percentiles(video_path: str, percentiles: range, max_frames: int = 1, min_area: int = 50):
    """
    Analyze the number of cells detected for different percentiles and collect labeled and original images for percentiles 90-95.
    
    Args:
        video_path (str): Path to the input video.
        percentiles (range): Range of percentiles to test (e.g., range(80, 100)).
        max_frames (int): Number of frames to process (default: 1).
        min_area (int): Minimum area for segmented objects (default: 50).
    
    Returns:
        tuple: (pd.DataFrame with 'percentile', 'num_cells', and 'video_name', dict of labeled images, np.ndarray of original image).
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
            results.append({
                'percentile': percentile,
                'num_cells': avg_num_cells,
                'video_name': Path(video_path).stem
            })
            logger.info(f"Average number of cells for percentile {percentile} in {Path(video_path).stem}: {avg_num_cells:.2f}")
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return pd.DataFrame(), {}, None
    
    return pd.DataFrame(results), labeled_images, original_image

def find_optimal_percentile(results: pd.DataFrame) -> float:
    """
    Find the optimal percentile based on the elbow point around 90 with a drop-off, aggregated across all videos.
    
    Args:
        results (pd.DataFrame): DataFrame with 'percentile', 'num_cells', and 'video_name' columns.
    
    Returns:
        float: Optimal percentile value.
    """
    if results.empty:
        logger.warning("No results to analyze for optimal percentile. Defaulting to 90.")
        return 90.0
    
    # Aggregate mean num_cells across videos for each percentile
    agg_results = results.groupby('percentile')['num_cells'].mean().reset_index()
    agg_results = agg_results.sort_values('percentile')
    num_cells = agg_results['num_cells'].values
    percentiles = agg_results['percentile'].values
    
    # Smooth the cell counts to reduce noise
    if len(num_cells) >= 3:
        num_cells_smoothed = savgol_filter(num_cells, window_length=3, polyorder=1)
    else:
        num_cells_smoothed = num_cells
    
    # Compute first differences to detect drop-off
    deltas = np.diff(num_cells_smoothed)
    
    # Find elbow around 90: prioritize drop-off after 90
    elbow_candidate = None
    for i in range(len(percentiles) - 1):
        if percentiles[i] >= 90 and deltas[i] < 0:  # Drop-off after 90
            elbow_candidate = percentiles[i]
            break
    
    if elbow_candidate is None:
        # Fallback to percentile near 90 with highest cell count
        idx_near_90 = np.argmin(np.abs(percentiles - 90))
        elbow_candidate = percentiles[idx_near_90]
    
    logger.info(f"Optimal percentile: {elbow_candidate} (mean num_cells: {num_cells[percentiles.tolist().index(elbow_candidate)]:.2f})")
    return float(elbow_candidate)

def plot_percentile_results(results: pd.DataFrame, output_path: str, dpi: int = 150):
    """
    Plot the number of cells versus percentile for each video with the optimal percentile highlighted.
    
    Args:
        results (pd.DataFrame): DataFrame with 'percentile', 'num_cells', and 'video_name' columns.
        output_path (str): Path to save the plot.
        dpi (int): Resolution of the output image (default: 150).
    """
    if results.empty:
        logger.warning("No results to plot.")
        return
    
    optimal_percentile = find_optimal_percentile(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results, x='percentile', y='num_cells', hue='video_name', marker='o')
    plt.axvline(x=optimal_percentile, color='red', linestyle='--', label=f'Optimal Percentile: {optimal_percentile}')
    plt.xlabel('Percentile')
    plt.ylabel('Average Number of Cells')
    plt.title('Number of Cells Detected vs. Percentile Across Videos')
    plt.grid(True)
    plt.legend(title='Video Name')
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved percentile plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save percentile plot to {output_path}: {e}")
    plt.close()

def plot_labeled_comparison(labeled_images: dict, original_images: dict, output_path: str, dpi: int = 150, fontsize: int = 6):
    """
    Plot labeled and original images for percentiles 90-95 for all videos in a grid layout.
    
    Args:
        labeled_images (dict): Dictionary mapping (video_name, percentile) to labeled images.
        original_images (dict): Dictionary mapping video_name to original images.
        output_path (str): Path to save the comparison figure.
        dpi (int): Resolution of the output image (default: 150).
        fontsize (int): Font size for cell ID annotations (default: 6).
    """
    if not labeled_images or not original_images:
        logger.warning("No labeled or original images provided for comparison plot.")
        return
    
    video_names = sorted(set(video_name for video_name, _ in labeled_images.keys()))
    num_videos = len(video_names)
    
    # Define subplot grid: 2 rows per video (labeled images, original image)
    plt.figure(figsize=(18, 4 * num_videos))
    
    for v_idx, video_name in enumerate(video_names):
        # Plot labeled images for percentiles 90-95
        for p_idx, percentile in enumerate(range(90, 96)):
            subplot_idx = v_idx * 12 + p_idx + 1  # 12 subplots per video (6 labeled + 6 for original)
            plt.subplot(num_videos, 12, subplot_idx)
            key = (video_name, percentile)
            if key not in labeled_images:
                logger.warning(f"No labeled image for {video_name} at percentile {percentile}. Skipping subplot.")
                plt.axis('off')
                continue
            labeled = labeled_images[key]
            num_cells = len(np.unique(labeled)) - 1
            plt.imshow(labeled, cmap='nipy_spectral')
            regions = regionprops(labeled)
            for region in regions:
                y, x = region.centroid
                plt.text(x, y, str(region.label), color='white', fontsize=fontsize, ha='center', va='center')
            plt.title(f'{video_name}\nPercentile {percentile} ({num_cells} cells)', fontsize=8)
            plt.axis('off')
        
        # Plot original image (spanning multiple columns for clarity)
        subplot_idx = v_idx * 12 + 7  # Start original image in 7th column
        plt.subplot(num_videos, 12, subplot_idx)
        if video_name in original_images:
            plt.imshow(original_images[video_name], cmap='gray')
            plt.title(f'{video_name}\nOriginal Image', fontsize=8)
            plt.axis('off')
        else:
            logger.warning(f"No original image for {video_name}. Skipping subplot.")
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
    """
    Main function to analyze percentiles across all videos, and determine the optimal percentile.
    """
    # Define video directories
    data_dir = project_root / 'data/raw'
    wildtype_dir = data_dir / 'wildtype'
    mutant_dir = data_dir / 'mutant'
    
    # Collect all video paths
    video_paths = []
    for directory in [wildtype_dir, mutant_dir]:
        for video_path in directory.glob('*.mp4'):
            video_paths.append(str(video_path))
    
    if not video_paths:
        logger.error("No videos found in wildtype or mutant directories.")
        raise FileNotFoundError("No videos found.")
    
    logger.info(f"Found {len(video_paths)} videos: {[Path(p).stem for p in video_paths]}")
    
    # Define percentile range
    percentiles = range(80, 100)
    
    # Analyze percentiles for all videos
    all_results = []
    all_labeled_images = {}
    all_original_images = {}
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        logger.info(f"Processing video: {video_name}")
        results, labeled_images, original_image = analyze_percentiles(video_path, percentiles, max_frames=1, min_area=50)
        if not results.empty:
            all_results.append(results)
            for percentile, labeled in labeled_images.items():
                all_labeled_images[(video_name, percentile)] = labeled
            if original_image is not None:
                all_original_images[video_name] = original_image
    
    # Combine results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        output_dir = project_root / 'results/figures/percentile_analysis'
        
        # Save combined results to CSV
        try:
            csv_path = output_dir / 'percentile_results_all_videos.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            combined_results.to_csv(csv_path, index=False)
            logger.info(f"Saved combined percentile results to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save percentile results to {csv_path}: {e}")
        
        # Plot results
        output_path = str(output_dir / 'percentile_vs_num_cells_all_videos.png')
        plot_percentile_results(combined_results, output_path)
        
        # Plot labeled comparison
        comparison_path = str(output_dir / 'percentile_comparison_90_95_all_videos.png')
        plot_labeled_comparison(all_labeled_images, all_original_images, comparison_path)
        
        # Find and log optimal percentile
        optimal_percentile = find_optimal_percentile(combined_results)
        logger.info(f"Global optimal percentile across all videos: {optimal_percentile}")

if __name__ == '__main__':
    main()