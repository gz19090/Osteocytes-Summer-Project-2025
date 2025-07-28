import sys
from pathlib import Path
import pandas as pd
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
import argparse
import logging
import numpy as np

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

from src.image_utils import load_video, correct_background, apply_fourier_filter, save_image
from src.segmentation import apply_edge_filters, segment_cells
from src.analysis import analyze_cells, analyze_dendrites
from src.visualization import plot_edge_filters, plot_combined_image, plot_contours, plot_segmentation, plot_histograms

parser.add_argument('--percentile', type=float, default=87, 
                    help='Percentile for thresholding if use_percentile=True (default: 87)')
                    
def main(max_frames: int = None, min_area: int = 10, use_percentile: bool = False, percentile: float = 87, crop: tuple = None):
    # Pass percentile to segment_cells
    labeled, _, contours = segment_cells(filtered, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
    labeled_cropped, _, contours_cropped = segment_cells(cropped, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
    """Process videos in wildtype and mutant subfolders, with optional parameters.
    
    Args:
        max_frames (int, optional): Maximum number of frames to process per video. If None or 0, process all frames.
        min_area (int): Minimum area for segmented objects.
        use_percentile (bool): Use percentile thresholding instead of Otsu.
        crop (tuple, optional): Crop region (y1, y2, x1, x2) or None for full image.
    """
    # If max_frames not provided, prompt user
    if max_frames is None:
        try:
            response = input("Enter number of frames to process (e.g., 10, or 'all' for all frames): ").strip().lower()
            max_frames = 0 if response == 'all' else int(response)
        except ValueError as e:
            logger.error(f"Invalid input for max_frames: {e}. Defaulting to all frames.")
            max_frames = 0
    
    # Log segmentation parameters
    logger.info(f"Segmentation parameters: min_area={min_area}, use_percentile={use_percentile}")
    if crop:
        logger.info(f"Crop region: {crop}")
    else:
        logger.info("No cropping applied.")
    
    # Paths
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    results_dir = Path('results/figures')
    metrics_dir = Path('results/metrics')
    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Process videos in wildtype and mutant subfolders
    for condition_dir in [data_dir / 'wildtype', data_dir / 'mutant']:
        if not condition_dir.exists():
            logger.warning(f"Directory {condition_dir} not found. Skipping.")
            continue
        
        condition = condition_dir.name  # 'wildtype' or 'mutant'
        condition_output_dir = output_dir / condition
        condition_results_dir = results_dir / condition
        condition_metrics_dir = metrics_dir / condition
        condition_output_dir.mkdir(exist_ok=True)
        condition_results_dir.mkdir(exist_ok=True)
        condition_metrics_dir.mkdir(exist_ok=True)
        
        for video_path in condition_dir.glob('*.mp4'):
            logger.info(f"Processing video: {video_path}")
            # Create video-specific subfolders
            video_name = video_path.stem
            video_output_dir = condition_output_dir / video_name
            video_results_dir = condition_results_dir / video_name
            video_output_dir.mkdir(exist_ok=True)
            video_results_dir.mkdir(exist_ok=True)
            
            # Load and preprocess video
            try:
                frames = load_video(str(video_path))
            except Exception as e:
                logger.error(f"Error loading {video_path}: {e}. Skipping.")
                continue
            
            # Limit frames if specified
            frames_to_process = frames if max_frames == 0 else frames[:max_frames]
            
            # Process each frame
            metrics = []
            for frame_idx, frame in enumerate(frames_to_process):
                logger.info(f"Processing frame {frame_idx} of {video_path.stem} ({condition})")
                try:
                    # Preprocessing
                    corrected = correct_background(frame)
                    filtered = apply_fourier_filter(corrected)
                    
                    # Process full image
                    combined, weights = apply_edge_filters(filtered)
                    labeled, _, contours = segment_cells(filtered, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
                    
                    # Process cropped image (if specified)
                    if crop:
                        y1, y2, x1, x2 = crop
                        if y1 < y2 <= filtered.shape[0] and x1 < x2 <= filtered.shape[1]:
                            cropped = filtered[y1:y2, x1:x2]
                            combined_cropped, weights_cropped = apply_edge_filters(cropped)
                            labeled_cropped, _, contours_cropped = segment_cells(cropped, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
                        else:
                            logger.warning(f"Invalid crop {crop} for frame {frame_idx}. Skipping cropped processing.")
                            cropped, combined_cropped, weights_cropped, labeled_cropped, contours_cropped = None, None, None, None, None
                    else:
                        cropped, combined_cropped, weights_cropped, labeled_cropped, contours_cropped = None, None, None, None, None
                    
                    # Analysis
                    cell_metrics = analyze_cells(labeled, filtered)
                    dendrite_metrics = analyze_dendrites(labeled > 0, index=cell_metrics.index)
                    cell_metrics['video'] = video_path.stem
                    cell_metrics['frame'] = frame_idx
                    cell_metrics['condition'] = condition
                    cell_metrics['dendritic_length'] = dendrite_metrics['dendritic_length']
                    metrics.append(cell_metrics)
                    
                    if cropped is not None:
                        cell_metrics_cropped = analyze_cells(labeled_cropped, cropped)
                        dendrite_metrics_cropped = analyze_dendrites(labeled_cropped > 0, index=cell_metrics_cropped.index)
                        cell_metrics_cropped['video'] = f'{video_path.stem}_cropped'
                        cell_metrics_cropped['frame'] = frame_idx
                        cell_metrics_cropped['condition'] = condition
                        cell_metrics_cropped['dendritic_length'] = dendrite_metrics_cropped['dendritic_length']
                        metrics.append(cell_metrics_cropped)
                    
                    # Save outputs
                    frame_prefix = f'{condition}_{video_path.stem}_frame_{frame_idx:04d}'
                    save_image(filtered, str(video_output_dir / f'{frame_prefix}_filtered.tif'))
                    save_image(combined, str(video_output_dir / f'{frame_prefix}_combined.tif'))
                    save_image(labeled, str(video_output_dir / f'{frame_prefix}_labeled.tif'))
                    plot_edge_filters(filtered, str(video_results_dir / f'{frame_prefix}_edge_filters.png'))
                    plot_combined_image(filtered, combined, weights, str(video_results_dir / f'{frame_prefix}_combined.png'))
                    plot_contours(combined, contours, str(video_results_dir / f'{frame_prefix}_contours.png'))
                    plot_segmentation(filtered, combined, labeled, str(video_results_dir / f'{frame_prefix}_segmentation.png'))
                    plot_histograms(filtered, cell_metrics['area'].tolist(), cell_metrics['dendritic_length'].tolist(), cell_metrics['eccentricity'].tolist(), cell_metrics['solidity'].tolist(), str(video_results_dir / f'{frame_prefix}_histograms.png'))
                    
                    if cropped is not None:
                        save_image(cropped, str(video_output_dir / f'{frame_prefix}_cropped.tif'))
                        save_image(combined_cropped, str(video_output_dir / f'{frame_prefix}_combined_cropped.tif'))
                        save_image(labeled_cropped, str(video_output_dir / f'{frame_prefix}_labeled_cropped.tif'))
                        plot_edge_filters(cropped, str(video_results_dir / f'{frame_prefix}_edge_filters_cropped.png'))
                        plot_combined_image(cropped, combined_cropped, weights_cropped, str(video_results_dir / f'{frame_prefix}_combined_cropped.png'))
                        plot_contours(combined_cropped, contours_cropped, str(video_results_dir / f'{frame_prefix}_contours_cropped.png'))
                        plot_segmentation(cropped, combined_cropped, labeled_cropped, str(video_results_dir / f'{frame_prefix}_segmentation_cropped.png'))
                        plot_histograms(cropped, cell_metrics_cropped['area'].tolist(), cell_metrics_cropped['dendritic_length'].tolist(), cell_metrics_cropped['eccentricity'].tolist(), cell_metrics_cropped['solidity'].tolist(), str(video_results_dir / f'{frame_prefix}_histograms_cropped.png'))
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx} of {video_path}: {e}")
                    continue
            
            # Save metrics for this video
            if metrics:
                try:
                    metrics_df = pd.concat(metrics, ignore_index=True)
                    metrics_path = condition_metrics_dir / f'{video_name}_metrics.csv'
                    metrics_df.to_csv(metrics_path, index=False)
                    logger.info(f"Metrics saved to {metrics_path}")
                except Exception as e:
                    logger.error(f"Error saving metrics for {video_path}: {e}")
            else:
                logger.warning(f"No metrics generated for {video_path}. Check video frames and processing steps.")
    
    logger.info("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process osteocyte videos with optional parameters.")
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Maximum number of frames to process per video (default: None, prompts user)')
    parser.add_argument('--min-area', type=int, default=10, 
                        help='Minimum area for segmented objects (default: 10)')
    parser.add_argument('--use-percentile', action='store_true', 
                        help='Use percentile thresholding instead of Otsu (default: False)')
    parser.add_argument('--percentile', type=float, default=87, 
                    help='Percentile for thresholding if use_percentile=True (default: 87)')
    parser.add_argument('--crop', type=int, nargs=4, default=None, 
                        help='Crop region as y1 y2 x1 x2 (default: None, no cropping)')
    args = parser.parse_args()
    main(max_frames=args.max_frames, min_area=args.min_area, use_percentile=args.use_percentile, crop=args.crop)