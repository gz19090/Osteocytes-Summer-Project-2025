# Osteocyte 2D Cell Culture Analysis
This project analyzes videos of 2D osteocyte cell cultures to segment and quantify cells and dendritic processes using parabolic FFT filtering and contour-based segmentation. It processes videos in `wildtype` and `mutant` conditions, generating metrics (cell area, intensity, eccentricity, dendrite count), per-frame timing profiles, and visualizations (edge filter plots, contour overlays, segmentation masks, histograms, skeleton overlays) for user-specified frames.

## Project Structure
```
Osteocytes culture/
├── src/
│   ├── __init__.py
│   ├── image_utils.py        # Functions for video loading, background correction, Fourier filtering, image saving
│   ├── segmentation.py       # Edge filter optimization and contour-based segmentation
│   ├── analysis.py           # Cell and dendritic process analysis
│   ├── visualization.py      # Plotting functions for edge filters, contours, segmentation, histograms with DPI control
├── scripts/
│   ├── main_workflow.py      # Main script to process videos/frames with parallel processing, subsampling, batch I/O, and DPI control
│   ├── analyze_percentiles.py # Analyzes cell counts across percentiles for all videos and generates comparison figures
├── notebooks/
│   ├── morphology.ipynb      # Analysis of morphological features comparing wildtype vs LTBP3-deficient (mutant)
│   ├── timing_analysis.ipynb # Analysis of processing time for each pipeline stage, comparing wildtype vs mutant
├── data/
│   ├── raw/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1.mp4
│   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1.mp4
│   │   │   ├── ...
│   ├── processed/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1/
│   │   │   │   ├── frame_0000/
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_filtered.tif
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.tif
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled.tif
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_cropped.tif
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined_cropped.tif
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled_cropped.tif
│   │   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│   │   │   │   ├── frame_0000/
│   │   │   │   │   ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_filtered.tif
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
├── results/
│   ├── figures/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1/
│   │   │   │   ├── frame_0000/
│   │   │   │   │   ├── skeletons/
│   │   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_skeleton_cell_1.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters.png
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.png
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_contours.png
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_segmentation.png
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_histograms.png
│   │   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters_cropped.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│   │   │   │   ├── frame_0000/
│   │   │   │   │   ├── skeletons/
│   │   │   │   │   │   ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_skeleton_cell_1.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_edge_filters.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   ├── percentile_analysis/
│   │   │   ├── percentile_results_all_videos.csv
│   │   │   ├── percentile_vs_num_cells_all_videos.png
│   │   │   ├── percentile_comparison_90_95_all_videos.png
│   │   ├── timing_analysis/
│   │   │   ├── speedup_comparison.png
│   │   │   ├── ...
│   ├── metrics/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1_metrics.csv
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1_timings.csv
│   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1_metrics.csv
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1_timings.csv
│   │   │   ├── ...
│   ├── morph_plots/
│   │   ├── correlation_heatmap.png
│   │   ├── histogram_plots.png
│   │   ├── ...
├── README.md
├── requirements.txt
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/gz19090/Osteocytes-Summer-Project-2025.git
   cd Osteocytes-Summer-Project-2025/Osteocytes culture
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place videos in `data/raw/wildtype/` and `data/raw/mutant/` in MP4 format.

## Scripts
### main_workflow.py
Processes all videos in `data/raw/wildtype/` and `data/raw/mutant/`, applying background correction, Fourier filtering, edge detection, and contour-based segmentation. Generates metrics (cell area, intensity, eccentricity, dendrite count) and visualizations (edge filters, contours, segmentation masks, histograms, skeleton overlays) for each frame. Optimized with frame-level parallel processing (4–8 processes), configurable frame subsampling, batch I/O, data reuse for cropping, and DPI control for visualizations.

**Usage**:
```bash
python scripts/main_workflow.py
```
- Prompts for the number of frames to process (e.g., "10" or "all").
- Alternatively, specify parameters via command-line:
  ```bash
  python scripts/main_workflow.py --max-frames 48 --min-area 50 --use-percentile --percentile 90 --crop 40 150 250 512 --num-wildtype 3 --num-mutant 3 --subsample-rate 1 --dpi 100
  ```
- `--max-frames`: Number of frames to process per video (default: None, prompts user).
- `--min-area`: Minimum area for segmented cells (default: 50).
- `--use-percentile`: Use percentile thresholding instead of Otsu (default: True).
- `--percentile`: Percentile for thresholding (default: 90).
- `--crop`: Crop region as `y1 y2 x1 x2` (default: 40 150 250 512).
- `--num-wildtype`/`--num-mutant`: Number of videos to process per condition (default: 3).
- `--subsample-rate`: Process every nth frame (default: 1, process all frames).
- `--dpi`: DPI for visualization plots (default: 100, lower values reduce visualization time).

**Outputs**:
- Processed images: `data/processed/<condition>/<video_name>/frame_XXXX/`
- Visualizations: `results/figures/<condition>/<video_name>/frame_XXXX/` (includes `skeletons/` subfolder for overlays comparing full skeletons vs dendrite skeletons used for counting).
- Metrics: `results/metrics/<condition>/<video_name>_metrics.csv`
- Per-frame timings: `results/metrics/<condition>/<video_name>_timings.csv` (columns: `frame_idx`, `video_name`, `condition`, `t_preprocess_s`, `t_segment_s`, `t_metrics_s`, `t_io_s`, `t_visualize_s`, `t_crop_total_s`, `t_total_s`, `has_crop`), compatible with `timing_analysis.ipynb`.

### analyze_percentiles.py
Analyzes cell segmentation across percentiles (80–99) for all videos in `data/raw/wildtype/` and `data/raw/mutant/` to determine the optimal percentile for thresholding. Processes the first frame of each video by default. Generates:
- A CSV file with cell counts per percentile for all videos (`percentile_results_all_videos.csv`).
- A plot of average cell counts versus percentile across videos, with the optimal percentile (elbow around 90) highlighted (`percentile_vs_num_cells_all_videos.png`).
- A grid comparison figure for percentiles 90–95, showing labeled cell images (with cell IDs) and original (filtered) images for each video (`percentile_comparison_90_95_all_videos.png`).

**Usage**:
```bash
python scripts/analyze_percentiles.py
```
- Processes all videos in `data/raw/wildtype/` and `data/raw/mutant/` (default: `max_frames=1`, `min_area=50`, `dpi=150` for plots).
- Outputs saved in `results/figures/percentile_analysis/`.

**Choosing the Optimal Percentile**:
- Inspect `percentile_comparison_90_95_all_videos.png` to compare labeled cells (top row per video) against the original image (bottom row per video) for percentiles 90–95 across all videos.
- Select the percentile with clear cell boundaries, minimal noise (small spurious regions), and complete cell detection.
- Update `main_workflow.py` to set `--percentile X` (e.g., 90) if a better value is identified.

## Usage
Run the full pipeline:
```bash
python scripts/main_workflow.py
```
For specific parameters, including subsampling and DPI control:
```bash
python scripts/main_workflow.py --max-frames 48 --min-area 50 --use-percentile --percentile 90 --crop 40 150 250 512 --num-wildtype 3 --num-mutant 3 --subsample-rate 1 --dpi 100
```
Analyze percentiles to optimize thresholding:
```bash
python scripts/analyze_percentiles.py
```
Explore morphological analysis:
```bash
cd /.../Osteocytes-Summer-Project-2025/Osteocytes culture
jupyter notebook notebooks/morphology.ipynb
```
Explore timing analysis of the pipeline:
```bash
cd /.../Osteocytes-Summer-Project-2025/Osteocytes culture
jupyter notebook notebooks/timing_analysis.ipynb
```

## Outputs
- Processed images and visualizations: `data/processed/` and `results/figures/`
- Metrics: `results/metrics/`
- Timing profiles (per frame): `results/metrics/<condition>/<video_name>_timings.csv`
- Morphological plots: `results/morph_plots/`
- Percentile analysis: `results/figures/percentile_analysis/`
- `percentile_results_all_videos.csv`: Cell counts per percentile for all videos.
- `percentile_vs_num_cells_all_videos.png`: Plot of average cell counts across videos with optimal percentile.
- `percentile_comparison_90_95_all_videos.png`: Grid comparison of labeled cells and original images for percentiles 90–95 across all videos.
- Timing analysis: `results/figures/timing_analysis/`
- `avg_time_per_stage.png`: Bar plot showing the average time for each processing stage.
- `speedup_comparison.png`: Bar plot comparing sequential and parallel processing times.
- `total_time_breakdown.png`: Stacked bar plot showing the total processing time for each video, split by stage.
- `total_time_per_frame.png`: Box plot showing the range of total processing times per frame for each video.

## Dependencies
See `requirements.txt`. Key libraries:
- `numpy==1.26.4`
- `scikit-image==0.22.0`
- `opencv-python`
- `matplotlib==3.9.0`
- `pandas>=2.2.0`
- `scipy>=1.14.0`
- `scikit-learn`
- `seaborn`
- `statsmodels`
- `networkx`
- `tqdm`
- `imageio==2.34.0`
- `imageio-ffmpeg`
- `plotly`
- `pytest`
- `joblib`
- `pyamg`

## Notes
- Videos must be in MP4 format and placed in `data/raw/wildtype/` or `data/raw/mutant/`.
- Adjust `min_area` in `segment_cells` (in `src/segmentation.py`) if too few cells are detected (e.g., `min_area=50`).
- The default percentile in `main_workflow.py` is 90, validated by `analyze_percentiles.py` across all videos. Run `analyze_percentiles.py` to confirm or select a better value.
- dendrite_count is computed from protrusion skeletons: if a skeleton has fewer than two branches, it is considered non-dendritic (0).
- For performance with many frames, set a smaller `max_frames` (e.g., 48) or increase `--subsample-rate` (e.g., 5 for every 5th frame). Parallel processing (4–8 processes), batch I/O, and DPI control improve runtime efficiency.

## Performance and Optimization

The osteocyte video analysis pipeline leverages frame-level parallel processing with 4–8 processes, batch I/O, data reuse for cropping, and DPI control for visualizations. For 6 videos (3 wildtype: `C2_3`, `C4_1`, `C5_2`; 3 mutant: `G2_1`, `G4_2`, `G5_1`) with 48 frames each (288 frames total), the pipeline achieves a total runtime of ~37 min (~6 min per video) with 8 processes. This represents a x8 speedup due to:
- **Batch I/O**: Reduces I/O time.
- **Data Reuse**: Reuses filtered images for cropping.
- **DPI Control**: Lowers visualization time.

Bar plot analysis (`timing_analysis.ipynb`) identifies Metrics, Visualize, and Crop as primary bottlenecks. 

The `analyze_percentiles.py` script analyzes percentiles 80–99 to identify the optimal thresholding percentile (around 90).

**Future Improvements**:
1. Further optimize visualization by reducing DPI (e.g., `--dpi 50`) or skipping non-essential plots (e.g., edge filters).
2. Enhance segmentation with GPU-accelerated algorithms using AWS EC2 instances.
3. Implement asynchronous I/O with `aiofiles` to minimize overhead.
4. Enable parallel video processing with careful resource management to avoid daemon errors in `miniforge3`.
5. Use adaptive subsampling or dynamic resource allocation to reduce variability.
6. Cache intermediate results to skip redundant computations.

## Quick Timing Summary (optional)
After a run, summarize average stage times in terminal (Python one-liner):
```bash
python - <<'PY'
import pandas as pd, glob
paths = glob.glob('results/metrics/*/*_timings.csv')
df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
cols = ['t_preprocess_s','t_segment_s','t_metrics_s','t_io_s','t_visualize_s','t_crop_total_s','t_total_s']
print(df.groupby('condition')[cols].mean().round(3))
PY
```