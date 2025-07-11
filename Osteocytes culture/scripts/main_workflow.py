from pathlib import Path
import pandas as pd
from skimage.filters import sobel, scharr, prewitt, roberts, farid, laplace
from sklearn.linear_model import LinearRegression
from src.image_utils import load_video, correct_background, apply_fourier_filter, save_image
from src.segmentation import apply_edge_filters, segment_cells
from src.analysis import analyze_cells, analyze_dendrites
from src.visualization import plot_edge_filters, plot_combined_image, plot_contours, plot_segmentation, plot_histograms
import argparse

def compute_edge_weights(image: np.ndarray) -> list:
    """Compute weights for edge filters.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        list: Weights for edge filters.
    """
    edge_filters = [sobel, scharr, prewitt, roberts, farid, laplace]
    filtered_imgs = [f(image).ravel() for f in edge_filters]
    X = np.stack(filtered_imgs, axis=1)
    y = image.ravel()
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    return reg.coef_

def main(max_frames: int = None):
    """Process videos in wildtype and mutant subfolders, with optional frame limit.
    
    Args:
        max_frames (int, optional): Maximum number of frames to process per video. If None or 0, process all frames.
    """
    # If max_frames not provided, prompt user
    if max_frames is None:
        response = input("Enter number of frames to process (e.g., 10, or 'all' for all frames): ").strip().lower()
        max_frames = 0 if response == 'all' else int(response)
    
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
            print(f"Directory {condition_dir} not found. Skipping.")
            continue
        
        condition = condition_dir.name  # 'wildtype' or 'mutant'
        condition_output_dir = output_dir / condition
        condition_results_dir = results_dir / condition
        condition_metrics_dir = metrics_dir / condition
        condition_output_dir.mkdir(exist_ok=True)
        condition_results_dir.mkdir(exist_ok=True)
        condition_metrics_dir.mkdir(exist_ok=True)
        
        for video_path in condition_dir.glob('*.mp4'):
            print(f"Processing video: {video_path}")
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
                print(f"Error loading {video_path}: {e}. Skipping.")
                continue
            
            # Limit frames if specified
            frames_to_process = frames if max_frames == 0 else frames[:max_frames]
            
            # Process each frame
            metrics = []
            for frame_idx, frame in enumerate(frames_to_process):
                print(f"Processing frame {frame_idx} of {video_path.stem} ({condition})")
                # Preprocessing
                corrected = correct_background(frame)
                filtered = apply_fourier_filter(corrected)
                
                # Process full image
                combined = apply_edge_filters(filtered)
                weights = compute_edge_weights(filtered)
                labeled, _, contours = segment_cells(combined)
                
                # Process cropped region
                crop = (40, 150, 250, 512)  # y1, y2, x1, x2
                cropped = filtered[crop[0]:crop[1], crop[2]:crop[3]]
                combined_cropped = apply_edge_filters(cropped)
                weights_cropped = compute_edge_weights(cropped)
                labeled_cropped, _, contours_cropped = segment_cells(combined_cropped)
                
                # Analysis
                cell_metrics = analyze_cells(labeled, filtered)
                dendrite_metrics = analyze_dendrites(labeled > 0, index=cell_metrics.index)
                cell_metrics['video'] = video_path.stem
                cell_metrics['frame'] = frame_idx
                cell_metrics['condition'] = condition
                cell_metrics['dendritic_length'] = dendrite_metrics['dendritic_length']
                metrics.append(cell_metrics)
                
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
                save_image(cropped, str(video_output_dir / f'{frame_prefix}_cropped.tif'))
                save_image(combined_cropped, str(video_output_dir / f'{frame_prefix}_combined_cropped.tif'))
                save_image(labeled_cropped, str(video_output_dir / f'{frame_prefix}_labeled_cropped.tif'))
                
                # Visualize
                plot_edge_filters(filtered, str(video_results_dir / f'{frame_prefix}_edge_filters.png'))
                plot_combined_image(filtered, combined, weights, str(video_results_dir / f'{frame_prefix}_combined.png'))
                plot_contours(combined, contours, str(video_results_dir / f'{frame_prefix}_contours.png'))
                plot_segmentation(filtered, combined, labeled, str(video_results_dir / f'{frame_prefix}_segmentation.png'))
                plot_histograms(filtered, cell_metrics['area'].tolist(), str(video_results_dir / f'{frame_prefix}_histograms.png'))
                
                plot_edge_filters(cropped, str(video_results_dir / f'{frame_prefix}_edge_filters_cropped.png'))
                plot_combined_image(cropped, combined_cropped, weights_cropped, str(video_results_dir / f'{frame_prefix}_combined_cropped.png'))
                plot_contours(combined_cropped, contours_cropped, str(video_results_dir / f'{frame_prefix}_contours_cropped.png'))
                plot_segmentation(cropped, combined_cropped, labeled_cropped, str(video_results_dir / f'{frame_prefix}_segmentation_cropped.png'))
                plot_histograms(cropped, cell_metrics_cropped['area'].tolist(), str(video_results_dir / f'{frame_prefix}_histograms_cropped.png'))
            
            # Save metrics for this video
            if metrics:
                metrics_df = pd.concat(metrics, ignore_index=True)
                metrics_path = condition_metrics_dir / f'{video_name}_metrics.csv'
                metrics_df.to_csv(metrics_path, index=False)
                print(f"Metrics saved to {metrics_path}")
    
    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process osteocyte videos with optional frame limit.")
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Maximum number of frames to process per video (default: None, prompts user)')
    args = parser.parse_args()
    main(max_frames=args.max_frames)