# main_workflow.py 

"""
main_workflow.py
----------------
This file orchestrates the processing pipeline for osteocyte cell video analysis in the osteocyte culture project. 
It coordinates loading and preprocessing of video frames, segmenting cells, calculating morphological metrics 
(e.g., area, dendrite count), and generating visualizations for wildtype and mutant videos. 
The pipeline supports configurable parameters for segmentation, cropping, and subsampling, producing metrics 
and figures saved to organized directories. Optimized with parallel processing (using 4-8 processes) and 
configurable frame subsampling (default: process all frames) for efficient execution.

Key tasks include:
- Loading videos and preprocessing frames (grayscale, noise reduction, background correction).
- Segmenting cells using edge detection and contour filling.
- Analyzing cell metrics and dendrite counts.
- Creating visualizations (e.g., edge filters, segmentation, skeleton overlays, histograms).
- Parallel processing of frames to improve runtime efficiency.
- Subsampling frames (configurable rate, default processes every frame) to balance speed 
and temporal resolution.

Used to process videos, generate results for analysis, and validate cell morphology differences. 
Supports command-line arguments for customizing processing, including maximum frames, subsampling rate, 
and cropping regions.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import logging
import numpy as np
from skimage.segmentation import relabel_sequential
import os
from multiprocessing import Pool

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the project root directory and add it to the Python path
project_root = Path('/Users/diana/Desktop/Osteocytes-Summer-Project-2025/Osteocytes culture')
sys.path.append(str(project_root))
logger.info(f"Script path: {Path(__file__).resolve()}")
logger.info(f"Project root: {project_root}")
logger.info(f"sys.path: {sys.path}")

# Check if the src directory exists
src_dir = project_root / 'src'
if not src_dir.exists():
    logger.error(f"src directory not found at {src_dir}. Please verify project structure.")
    raise FileNotFoundError(f"src directory not found at {src_dir}")

# Import functions from other modules
from src.image_utils import load_video, correct_background, apply_fourier_filter, save_image
from src.segmentation import apply_edge_filters, segment_cells
from src.analysis import analyze_cells, analyze_dendrite_count
from src.visualization import plot_edge_filters, plot_combined_image, plot_contours, plot_segmentation, plot_histograms, plot_skeleton_overlays

def process_frame(args):
    """
    Process a single frame, including preprocessing, segmentation, metric calculation, and visualization.
    Designed to be run in parallel with multiprocessing.
    
    Args:
        args: Tuple containing (frame_idx, frame, video_path, condition, min_area, use_percentile, percentile, crop,
              frame_dir, frame_results_dir, skeleton_dir)
    
    Returns:
        Tuple of (cell_metrics, cell_metrics_cropped) DataFrames or (None, None) if processing fails.
    """
    frame_idx, frame, video_path, condition, min_area, use_percentile, percentile, crop, frame_dir, frame_results_dir, skeleton_dir = args
    try:
        logger.info(f"Processing frame {frame_idx} of {video_path.stem} ({condition})")
        # Preprocess the frame
        corrected = correct_background(frame)
        filtered = apply_fourier_filter(corrected)
        # Apply edge detection and segment cells
        combined, weights = apply_edge_filters(filtered)
        labeled, _, contours = segment_cells(
            filtered, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
        labeled, _, _ = relabel_sequential(labeled)
        # Calculate cell metrics
        cell_metrics = analyze_cells(labeled, filtered)
        # Count dendrites
        dendrite_metrics = analyze_dendrite_count(labeled, index=cell_metrics.index, output_dir=str(skeleton_dir))
        # Add metadata
        cell_metrics = cell_metrics.rename(columns={'label': 'cell_id'})
        cell_metrics['frame_idx'] = frame_idx
        cell_metrics['condition'] = condition
        cell_metrics['video_name'] = video_path.stem
        cell_metrics['dendrite_count'] = dendrite_metrics['dendrite_count'].values
        cell_metrics['is_dendritic'] = cell_metrics['dendrite_count'] > 0
        # Save images
        frame_prefix = f'{condition}_{video_path.stem}_frame_{frame_idx:04d}'
        save_image(filtered, str(frame_dir / f'{frame_prefix}_filtered.tif'))
        save_image(combined, str(frame_dir / f'{frame_prefix}_combined.tif'))
        save_image(labeled, str(frame_dir / f'{frame_prefix}_labeled.tif'))
        # Create visualizations
        plot_skeleton_overlays(labeled, combined, cell_metrics, str(skeleton_dir), image_original=filtered)
        plot_edge_filters(filtered, str(frame_results_dir / f'{frame_prefix}_edge_filters.png'))
        plot_combined_image(filtered, combined, weights, str(frame_results_dir / f'{frame_prefix}_combined.png'))
        plot_contours(combined, contours, str(frame_results_dir / f'{frame_prefix}_contours.png'))
        plot_segmentation(filtered, combined, labeled, str(frame_results_dir / f'{frame_prefix}_segmentation.png'))
        plot_histograms(
            filtered, cell_metrics['area'].tolist(), cell_metrics['dendrite_count'].tolist(),
            cell_metrics['eccentricity'].tolist(), cell_metrics['solidity'].tolist(),
            str(frame_results_dir / f'{frame_prefix}_histograms.png'))
        # Process cropped region if specified
        if crop:
            y1, y2, x1, x2 = crop
            if y1 < y2 <= filtered.shape[0] and x1 < x2 <= filtered.shape[1]:
                cropped = filtered[y1:y2, x1:x2]
                combined_cropped, weights_cropped = apply_edge_filters(cropped)
                labeled_cropped, _, contours_cropped = segment_cells(
                    cropped, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None)
                labeled_cropped, _, _ = relabel_sequential(labeled_cropped)
                cell_metrics_cropped = analyze_cells(labeled_cropped, cropped)
                dendrite_metrics_cropped = analyze_dendrite_count(
                    labeled_cropped, index=cell_metrics_cropped.index, output_dir=str(skeleton_dir))
                cell_metrics_cropped = cell_metrics_cropped.rename(columns={'label': 'cell_id'})
                cell_metrics_cropped['frame_idx'] = frame_idx
                cell_metrics_cropped['condition'] = condition
                cell_metrics_cropped['video_name'] = f'{video_path.stem}_cropped'
                cell_metrics_cropped['dendrite_count'] = dendrite_metrics_cropped['dendrite_count']
                cell_metrics_cropped['is_dendritic'] = cell_metrics_cropped['dendrite_count'] > 0
                plot_skeleton_overlays(labeled_cropped, combined_cropped, cell_metrics_cropped, str(skeleton_dir), image_original=cropped)
                save_image(cropped, str(frame_dir / f'{frame_prefix}_cropped.tif'))
                save_image(combined_cropped, str(frame_dir / f'{frame_prefix}_combined_cropped.tif'))
                save_image(labeled_cropped, str(frame_dir / f'{frame_prefix}_labeled_cropped.tif'))
                plot_edge_filters(cropped, str(frame_results_dir / f'{frame_prefix}_edge_filters_cropped.png'))
                plot_combined_image(cropped, combined_cropped, weights_cropped,
                                   str(frame_results_dir / f'{frame_prefix}_combined_cropped.png'))
                plot_contours(combined_cropped, contours_cropped,
                              str(frame_results_dir / f'{frame_prefix}_contours_cropped.png'))
                plot_segmentation(cropped, combined_cropped, labeled_cropped,
                                 str(frame_results_dir / f'{frame_prefix}_segmentation_cropped.png'))
                plot_histograms(
                    cropped, cell_metrics_cropped['area'].tolist(),
                    cell_metrics_cropped['dendrite_count'].tolist(),
                    cell_metrics_cropped['eccentricity'].tolist(),
                    cell_metrics_cropped['solidity'].tolist(),
                    str(frame_results_dir / f'{frame_prefix}_histograms_cropped.png'))
                return cell_metrics, cell_metrics_cropped
            else:
                logger.warning(f"Invalid crop {crop} for frame {frame_idx}. Skipping cropped processing.")
                return cell_metrics, None
        return cell_metrics, None
    except Exception as e:
        logger.error(f"Error processing frame {frame_idx} of {video_path}: {e}")
        return None, None

def main(max_frames: int = None, min_area: int = 10, use_percentile: bool = False, percentile: float = 94.0,
         crop: tuple = None, num_wildtype: int = None, num_mutant: int = None, subsample_rate: int = 1):
    """
    Run the full pipeline to process osteocyte videos and generate analysis results.
    This function loads videos, preprocesses frames, segments cells, calculates metrics, and creates visualizations
    for wildtype and mutant videos, saving results to organized folders. Uses parallel processing and subsampling for efficiency.

    Args:
        max_frames (int, optional): Maximum number of frames to process per video (None prompts user).
        min_area (int): Minimum cell area to keep during segmentation (default: 10 pixels).
        use_percentile (bool): Use percentile thresholding instead of Otsu (default: False).
        percentile (float): Percentile value for thresholding if use_percentile=True (default: 94.0).
        crop (tuple, optional): Region to crop (y1, y2, x1, x2) or None.
        num_wildtype (int, optional): Number of wildtype videos to process (None for all).
        num_mutant (int, optional): Number of mutant videos to process (None for all).
        subsample_rate (int): Process every nth frame (default: 1, process all frames).
    """
    # Validate subsample_rate
    if subsample_rate <= 0:
        logger.warning(f"Invalid subsample_rate {subsample_rate}. Must be positive. Defaulting to 1 (all frames).")
        subsample_rate = 1

    # Prompt user for max_frames if not provided
    if max_frames is None:
        try:
            response = input("Enter number of frames to process (e.g., 10, or 'all' for all frames): ").strip().lower()
            max_frames = 0 if response == 'all' else int(response)
        except ValueError as e:
            logger.error(f"Invalid input for max_frames: {e}. Defaulting to all frames.")
            max_frames = 0

    # Log the parameters used
    logger.info(f"Segmentation parameters: min_area={min_area}, use_percentile={use_percentile}, percentile={percentile}, subsample_rate={subsample_rate}")
    if subsample_rate == 1:
        logger.info("Processing all frames (no subsampling).")
    else:
        logger.info(f"Processing every {subsample_rate}th frame.")
    if crop:
        logger.info(f"Crop region: {crop}")
    else:
        logger.info("No cropping applied.")

    # Set up directories for data, processed images, figures, and metrics
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    results_dir = Path('results/figures')
    metrics_dir = Path('results/metrics')
    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    # Process videos in wildtype and mutant folders
    for condition_dir in [data_dir / 'wildtype', data_dir / 'mutant']:
        if not condition_dir.exists():
            logger.warning(f"Directory {condition_dir} not found. Skipping.")
            continue
        condition = condition_dir.name  # 'wildtype' or 'mutant'

        # Create condition-specific output folders
        condition_output_dir = output_dir / condition
        condition_results_dir = results_dir / condition
        condition_metrics_dir = metrics_dir / condition
        condition_output_dir.mkdir(exist_ok=True)
        condition_results_dir.mkdir(exist_ok=True)
        condition_metrics_dir.mkdir(exist_ok=True)

        # Get list of video files and limit based on num_wildtype or num_mutant
        video_paths = sorted(condition_dir.glob('*.mp4'))
        if condition == 'wildtype' and num_wildtype is not None:
            video_paths = video_paths[:num_wildtype]
            logger.info(f"Limiting to first {num_wildtype} wildtype videos.")
        elif condition == 'mutant' and num_mutant is not None:
            video_paths = video_paths[:num_mutant]
            logger.info(f"Limiting to first {num_mutant} mutant videos.")

        # Process each video
        for video_path in video_paths:
            logger.info(f"Processing video: {video_path}")
            # Create video-specific folders
            video_name = video_path.stem
            video_output_dir = condition_output_dir / video_name
            video_results_dir = condition_results_dir / video_name
            video_output_dir.mkdir(exist_ok=True)
            video_results_dir.mkdir(exist_ok=True)

            # Load and preprocess the video
            try:
                frames = load_video(str(video_path))
            except Exception as e:
                logger.error(f"Error loading {video_path}: {e}. Skipping.")
                continue

            # Limit the number of frames to process and apply subsampling
            frames_to_process = frames if max_frames == 0 else frames[:max_frames]
            frames_to_process = frames_to_process[::subsample_rate]
            logger.info(f"Processing {len(frames_to_process)} frames for {video_path.stem}")

            # Store metrics for all frames
            metrics = []

            # Prepare arguments for parallel processing
            frame_args = [
                (
                    frame_idx * subsample_rate,  # Adjust frame_idx to reflect original frame number
                    frame, video_path, condition, min_area, use_percentile, percentile, crop,
                    video_output_dir / f'frame_{frame_idx * subsample_rate:04d}',
                    video_results_dir / f'frame_{frame_idx * subsample_rate:04d}',
                    video_results_dir / f'frame_{frame_idx * subsample_rate:04d}' / 'skeletons'
                )
                for frame_idx, frame in enumerate(frames_to_process)
            ]

            # Create directories before parallel processing to avoid race conditions
            for args in frame_args:
                args[8].mkdir(exist_ok=True)  # frame_dir
                args[9].mkdir(exist_ok=True)  # frame_results_dir
                args[10].mkdir(exist_ok=True)  # skeleton_dir

            # Parallel processing with 4-8 processes
            num_processes = min(os.cpu_count(), 8)  # Use up to 8 processes or CPU count
            logger.info(f"Using {num_processes} processes for parallel frame processing")
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_frame, frame_args)

            # Collect results
            for cell_metrics, cell_metrics_cropped in results:
                if cell_metrics is not None:
                    metrics.append(cell_metrics)
                if cell_metrics_cropped is not None:
                    metrics.append(cell_metrics_cropped)

            # Save all metrics for the video
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process osteocyte videos with optional parameters.")
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process per video (default: None, prompts user)')
    parser.add_argument('--min-area', type=int, default=10,
                        help='Minimum area for segmented objects (default: 10)')
    parser.add_argument('--use-percentile', action='store_true',
                        help='Use percentile thresholding instead of Otsu (default: False)')
    parser.add_argument('--percentile', type=float, default=94,
                        help='Percentile for thresholding if use_percentile=True (default: 94)')
    parser.add_argument('--crop', type=int, nargs=4, default=None,
                        help='Crop region as y1 y2 x1 x2 (default: None, no cropping)')
    parser.add_argument('--num-wildtype', type=int, default=None,
                        help='Number of wildtype videos to process (default: None, all)')
    parser.add_argument('--num-mutant', type=int, default=None,
                        help='Number of mutant videos to process (default: None, all)')
    parser.add_argument('--subsample-rate', type=int, default=1,
                        help='Process every nth frame (default: 1, process all frames)')
    args = parser.parse_args()

    # Run the main function with command-line arguments
    main(max_frames=args.max_frames, min_area=args.min_area, use_percentile=args.use_percentile,
         percentile=args.percentile, crop=args.crop, num_wildtype=args.num_wildtype,
         num_mutant=args.num_mutant, subsample_rate=args.subsample_rate)