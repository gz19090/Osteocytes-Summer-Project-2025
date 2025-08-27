# main_workflow.py 

"""
main_workflow.py
----------------
This file orchestrates the processing pipeline for osteocyte cell video analysis
in the osteocyte culture project. It coordinates loading and preprocessing of
video frames, segmenting cells, calculating morphological metrics (e.g., area,
dendrite count), and generating visualizations for wildtype and mutant videos.
The pipeline supports configurable parameters for segmentation, cropping, and
subsampling, producing metrics and figures saved to organized directories.
Optimized with parallel processing (using 4–8 processes) and configurable frame
subsampling (default: process all frames) for efficiency.
Per-frame timing profiling is included. For each frame the timings are recorded
for preprocessing, segmentation, metrics, I/O, visualization, and optional crop.
Timings are saved as CSV in results/metrics for analysis and bottleneck detection.
"""
import sys
from pathlib import Path
import pandas as pd
import argparse
import logging
import numpy as np
from skimage.segmentation import relabel_sequential
import os
import time
from multiprocessing import Pool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Helper for timing
def _now():
    return time.perf_counter()

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
from src.visualization import (
    plot_edge_filters, plot_combined_image, plot_contours, plot_segmentation,
    plot_histograms, plot_skeleton_overlays
)

def process_frame(args):
    """
    Process a single frame, including preprocessing, segmentation, metric calculation,
    visualization, and timing with batch I/O and data reuse.
    
    Args:
        args: Tuple containing (frame_idx, frame, video_path, condition, min_area,
              use_percentile, percentile, crop, frame_dir, frame_results_dir, skeleton_dir, dpi)
    
    Returns:
        Tuple of (cell_metrics, cell_metrics_cropped, frame_timing) or (None, None, None) if processing fails.
    """
    (frame_idx, frame, video_path, condition, min_area, use_percentile, percentile, crop,
     frame_dir, frame_results_dir, skeleton_dir, dpi) = args
    try:
        logger.info(f"Processing frame {frame_idx} of {video_path.stem} ({condition})")
        t0 = _now()
        # --- Preprocessing ---
        t_pre0 = _now()
        corrected = correct_background(frame)
        filtered = apply_fourier_filter(corrected)
        t_pre1 = _now()
        # --- Segmentation ---
        t_seg0 = _now()
        combined, weights = apply_edge_filters(filtered)
        labeled, _, contours = segment_cells(
            filtered, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None
        )
        labeled, _, _ = relabel_sequential(labeled)
        t_seg1 = _now()
        # --- Metrics ---
        t_metrics0 = _now()
        cell_metrics = analyze_cells(labeled, filtered)
        dendrite_metrics = analyze_dendrite_count(
            labeled, index=cell_metrics.index, output_dir=str(skeleton_dir)
        )
        t_metrics1 = _now()
        # Add metadata
        cell_metrics = cell_metrics.rename(columns={'label': 'cell_id'})
        cell_metrics['frame_idx'] = frame_idx
        cell_metrics['condition'] = condition
        cell_metrics['video_name'] = video_path.stem
        cell_metrics['dendrite_count'] = dendrite_metrics['dendrite_count'].values
        cell_metrics['is_dendritic'] = cell_metrics['dendrite_count'] > 0
        # --- I/O: collect images for batch saving ---
        t_io0 = _now()
        frame_prefix = f'{condition}_{video_path.stem}_frame_{frame_idx:04d}'
        images_to_save = [
            (filtered, str(frame_dir / f'{frame_prefix}_filtered.tif')),
            (combined, str(frame_dir / f'{frame_prefix}_combined.tif')),
            (labeled, str(frame_dir / f'{frame_prefix}_labeled.tif'))
        ]
        t_io1 = _now()
        # --- Visualizations (with DPI) ---
        t_viz0 = _now()
        plot_skeleton_overlays(labeled, combined, cell_metrics, str(skeleton_dir), image_original=filtered, dpi=dpi)
        plot_edge_filters(filtered, str(frame_results_dir / f'{frame_prefix}_edge_filters.png'), dpi=dpi)
        plot_combined_image(filtered, combined, weights, str(frame_results_dir / f'{frame_prefix}_combined.png'), dpi=dpi)
        plot_contours(combined, contours, str(frame_results_dir / f'{frame_prefix}_contours.png'), dpi=dpi)
        plot_segmentation(filtered, combined, labeled, str(frame_results_dir / f'{frame_prefix}_segmentation.png'), dpi=dpi)
        plot_histograms(
            filtered, cell_metrics['area'].tolist(), cell_metrics['dendrite_count'].tolist(),
            cell_metrics['eccentricity'].tolist(), cell_metrics['solidity'].tolist(),
            str(frame_results_dir / f'{frame_prefix}_histograms.png'), dpi=dpi
        )
        t_viz1 = _now()
        # --- Cropped processing (optional, reuse filtered image) ---
        cell_metrics_cropped = None
        t_crop_total = 0.0
        images_to_save_cropped = []
        if crop:
            y1, y2, x1, x2 = crop
            if y1 < y2 <= filtered.shape[0] and x1 < x2 <= filtered.shape[1]:
                t_crop0 = _now()
                cropped = filtered[y1:y2, x1:x2]
                combined_cropped, weights_cropped = apply_edge_filters(cropped)
                labeled_cropped, _, contours_cropped = segment_cells(
                    cropped, min_area=min_area, use_percentile=use_percentile, percentile=percentile, crop=None
                )
                labeled_cropped, _, _ = relabel_sequential(labeled_cropped)
                cell_metrics_cropped = analyze_cells(labeled_cropped, cropped)
                dendrite_metrics_cropped = analyze_dendrite_count(
                    labeled_cropped, index=cell_metrics_cropped.index, output_dir=str(skeleton_dir)
                )
                cell_metrics_cropped = cell_metrics_cropped.rename(columns={'label': 'cell_id'})
                cell_metrics_cropped['frame_idx'] = frame_idx
                cell_metrics_cropped['condition'] = condition
                cell_metrics_cropped['video_name'] = f'{video_path.stem}_cropped'
                cell_metrics_cropped['dendrite_count'] = dendrite_metrics_cropped['dendrite_count']
                cell_metrics_cropped['is_dendritic'] = cell_metrics_cropped['dendrite_count'] > 0
                images_to_save_cropped = [
                    (cropped, str(frame_dir / f'{frame_prefix}_cropped.tif')),
                    (combined_cropped, str(frame_dir / f'{frame_prefix}_combined_cropped.tif')),
                    (labeled_cropped, str(frame_dir / f'{frame_prefix}_labeled_cropped.tif'))
                ]
                plot_skeleton_overlays(labeled_cropped, combined_cropped, cell_metrics_cropped, str(skeleton_dir), image_original=cropped, dpi=dpi)
                plot_edge_filters(cropped, str(frame_results_dir / f'{frame_prefix}_edge_filters_cropped.png'), dpi=dpi)
                plot_combined_image(cropped, combined_cropped, weights_cropped,
                                   str(frame_results_dir / f'{frame_prefix}_combined_cropped.png'), dpi=dpi)
                plot_contours(combined_cropped, contours_cropped,
                              str(frame_results_dir / f'{frame_prefix}_contours_cropped.png'), dpi=dpi)
                plot_segmentation(cropped, combined_cropped, labeled_cropped,
                                 str(frame_results_dir / f'{frame_prefix}_segmentation_cropped.png'), dpi=dpi)
                plot_histograms(
                    cropped, cell_metrics_cropped['area'].tolist(),
                    cell_metrics_cropped['dendrite_count'].tolist(),
                    cell_metrics_cropped['eccentricity'].tolist(),
                    cell_metrics_cropped['solidity'].tolist(),
                    str(frame_results_dir / f'{frame_prefix}_histograms_cropped.png'), dpi=dpi
                )
                t_crop1 = _now()
                t_crop_total = t_crop1 - t_crop0
            else:
                logger.warning(f"Invalid crop {crop} for frame {frame_idx}. Skipping cropped processing.")
        t1 = _now()
        # Batch save images
        for img, path in images_to_save + images_to_save_cropped:
            save_image(img, path)
        frame_timing = {
            'frame_idx': frame_idx,
            'video_name': video_path.stem,
            'condition': condition,
            't_preprocess_s': t_pre1 - t_pre0,
            't_segment_s': t_seg1 - t_seg0,
            't_metrics_s': t_metrics1 - t_metrics0,
            't_io_s': t_io1 - t_io0,
            't_visualize_s': t_viz1 - t_viz0,
            't_crop_total_s': t_crop_total,
            't_total_s': t1 - t0,
            'has_crop': bool(crop and t_crop_total > 0)
        }
        return cell_metrics, cell_metrics_cropped, frame_timing
    except Exception as e:
        logger.error(f"Error processing frame {frame_idx} of {video_path}: {e}")
        return None, None, None

def main(max_frames: int = None, min_area: int = 20, use_percentile: bool = True, percentile: float = 90.0,
         crop: tuple = (40, 150, 250, 512), num_wildtype: int = 3, num_mutant: int = 3, 
         subsample_rate: int = 1, dpi: int = 100):
    """
    Run the full pipeline to process osteocyte videos and generate analysis results.
    Uses frame-level parallel processing, batch I/O, data reuse, and DPI control.
    Saves metrics and per-frame timings compatible with existing analysis notebook.
    
    Args:
        max_frames (int): Maximum number of frames to process per video (None prompts user).
        min_area (int): Minimum area for segmented objects.
        use_percentile (bool): Use percentile thresholding.
        percentile (float): Percentile for thresholding (default: 90).
        crop (tuple): Crop region as (y1, y2, x1, x2) or None.
        num_wildtype (int): Number of wildtype videos to process (default: 3).
        num_mutant (int): Number of mutant videos to process (default: 3).
        subsample_rate (int): Process every nth frame (default: 1).
        dpi (int): DPI for visualization plots (default: 100).
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
    logger.info(f"Segmentation parameters: min_area={min_area}, use_percentile={use_percentile}, "
                f"percentile={percentile}, subsample_rate={subsample_rate}, dpi={dpi}")
    if subsample_rate == 1:
        logger.info("Processing all frames (no subsampling).")
    else:
        logger.info(f"Processing every {subsample_rate}th frame.")
    if crop:
        logger.info(f"Crop region: {crop}")
    else:
        logger.info("No cropping applied.")
    
    # Set up directories
    data_dir = project_root / 'data/raw'
    output_dir = project_root / 'data/processed'
    results_dir = project_root / 'results/figures'
    metrics_dir = project_root / 'results/metrics'
    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Process wildtype and mutant
    for condition_dir in [data_dir / 'wildtype', data_dir / 'mutant']:
        if not condition_dir.exists():
            logger.warning(f"Directory {condition_dir} not found. Skipping.")
            continue
        condition = condition_dir.name
        # Create output dirs
        condition_output_dir = output_dir / condition
        condition_results_dir = results_dir / condition
        condition_metrics_dir = metrics_dir / condition
        condition_output_dir.mkdir(exist_ok=True)
        condition_results_dir.mkdir(exist_ok=True)
        condition_metrics_dir.mkdir(exist_ok=True)
        # Videos
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
            # Subsampling
            frames_to_process = frames if max_frames == 0 else frames[:max_frames]
            frames_to_process = frames_to_process[::subsample_rate]
            logger.info(f"Processing {len(frames_to_process)} frames for {video_path.stem}")
            # Prepare args
            frame_args = [
                (
                    frame_idx * subsample_rate, frame, video_path, condition, min_area,
                    use_percentile, percentile, crop,
                    video_output_dir / f'frame_{frame_idx * subsample_rate:04d}',
                    video_results_dir / f'frame_{frame_idx * subsample_rate:04d}',
                    video_results_dir / f'frame_{frame_idx * subsample_rate:04d}' / 'skeletons',
                    dpi
                )
                for frame_idx, frame in enumerate(frames_to_process)
            ]
            # Create directories before parallel processing
            for args in frame_args:
                args[8].mkdir(exist_ok=True)
                args[9].mkdir(exist_ok=True)
                args[10].mkdir(exist_ok=True)
            # Parallel processing
            num_processes = min(os.cpu_count(), 8)  # Adjustable to 4 if needed
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_frame, frame_args)
            # Collect results
            metrics = []
            timings = []
            for res in results:
                if res is None:
                    continue
                cell_metrics, cell_metrics_cropped, frame_timing = res
                if cell_metrics is not None:
                    metrics.append(cell_metrics)
                if cell_metrics_cropped is not None:
                    metrics.append(cell_metrics_cropped)
                if frame_timing is not None:
                    timings.append(frame_timing)
            # Save metrics
            if metrics:
                try:
                    metrics_df = pd.concat(metrics, ignore_index=True)
                    metrics_path = condition_metrics_dir / f'{video_name}_metrics.csv'
                    metrics_df.to_csv(metrics_path, index=False)
                    logger.info(f"Metrics saved to {metrics_path}")
                except Exception as e:
                    logger.error(f"Error saving metrics for {video_path}: {e}")
            else:
                logger.warning(f"No metrics generated for {video_path}.")
            # Save timings
            if timings:
                try:
                    timings_df = pd.DataFrame(timings)
                    timings_path = condition_metrics_dir / f'{video_name}_timings.csv'
                    timings_df.to_csv(timings_path, index=False)
                    summary = timings_df[['t_preprocess_s', 't_segment_s', 't_metrics_s', 't_visualize_s', 't_io_s', 't_total_s']].mean()
                    logger.info(
                        f"[Timing] {video_name} avg/frame (s): "
                        f"pre={summary['t_preprocess_s']:.3f}, seg={summary['t_segment_s']:.3f}, "
                        f"metrics={summary['t_metrics_s']:.3f}, viz={summary['t_visualize_s']:.3f}, "
                        f"io={summary['t_io_s']:.3f}, total={summary['t_total_s']:.3f}"
                    )
                except Exception as e:
                    logger.error(f"Error saving timings for {video_path}: {e}")
    logger.info("Processing complete.")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process osteocyte videos with improved frame-level parallelism.")
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process per video (default: None, prompts user)')
    parser.add_argument('--min-area', type=int, default=20,
                        help='Minimum area for segmented objects (default: 20)')
    parser.add_argument('--use-percentile', action='store_true',
                        help='Use percentile thresholding instead of Otsu (default: False)')
    parser.add_argument('--percentile', type=float, default=90,
                        help='Percentile for thresholding if use_percentile=True (default: 90)')
    parser.add_argument('--crop', type=int, nargs=4, default=(40, 150, 250, 512),
                        help='Crop region as y1 y2 x1 x2 (default: 40 150 250 512)')
    parser.add_argument('--num-wildtype', type=int, default=3,
                        help='Number of wildtype videos to process (default: 3)')
    parser.add_argument('--num-mutant', type=int, default=3,
                        help='Number of mutant videos to process (default: 3)')
    parser.add_argument('--subsample-rate', type=int, default=1,
                        help='Process every nth frame (default: 1, process all frames)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for visualization plots (default: 100)')
    args = parser.parse_args()
    
    # Run the main function with command-line arguments
    main(max_frames=args.max_frames, min_area=args.min_area, use_percentile=args.use_percentile,
         percentile=args.percentile, crop=args.crop, num_wildtype=args.num_wildtype,
         num_mutant=args.num_mutant, subsample_rate=args.subsample_rate, dpi=args.dpi)