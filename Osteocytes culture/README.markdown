# Osteocyte 2D Cell Culture Analysis
This project analyzes videos of 2D osteocyte cell cultures to segment and quantify cells and dendritic processes using parabolic FFT filtering and contour-based segmentation. It processes videos in `wildtype` and `mutant` conditions, generating metrics (cell area, intensity, eccentricity, dendrite count), per-frame timing profiles, and visualizations (edge filter plots, contour overlays, segmentation masks, histograms, skeleton overlays) for user-specified frames.

## Project Structure
```
Osteocytes culture/
├── src/
│ ├── __init__.py
│ ├── image_utils.py # Functions for video loading, background correction, Fourier filtering, image saving
│ ├── segmentation.py # Edge filter optimization and contour-based segmentation
│ ├── analysis.py # Cell and dendritic process analysis
│ ├── visualization.py # Plotting functions for edge filters, contours, segmentation, histograms
├── scripts/
│ ├── main_workflow.py # Main script to process all videos and frames with parallel processing and subsampling
│ ├── analyze_percentiles.py # Analyzes cell counts across percentiles and generates comparison figures
├── notebooks/
│ ├── morphology.ipynb # Analysis of morphological features comparing control (wildtype) vs LTBP3-deficient (mutant)
│ ├── timing_analysis.ipynb # Analysis of the processing time for each stage of the osteocyte video analysis pipeline comparing performance between wildtype and LTBP3-deficient (mutant) cells.
├── data/
│ ├── raw/
│ │ ├── wildtype/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1.mp4
│ │ │ ├── ...
│ │ ├── mutant/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1.mp4
│ │ │ ├── ...
│ ├── processed/
│ │ ├── wildtype/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1/
│ │ │ │ ├── frame_0000/
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_filtered.tif
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.tif
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled.tif
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_cropped.tif
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined_cropped.tif
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled_cropped.tif
│ │ │ │ ├── frame_0001/
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0001_filtered.tif
│ │ │ │ │ ├── ...
│ │ │ │ ├── ...
│ │ │ ├── ...
│ │ ├── mutant/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│ │ │ │ ├── frame_0000/
│ │ │ │ │ ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_filtered.tif
│ │ │ │ │ ├── ...
│ │ │ │ ├── ...
│ │ │ ├── ...
├── results/
│ ├── figures/
│ │ ├── wildtype/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1/
│ │ │ │ ├── frame_0000/
│ │ │ │ │ ├── skeletons/
│ │ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_skeleton_cell_1.png
│ │ │ │ │ │ ├── ...
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_preprocessing.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_contours.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_segmentation.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_histograms.png
│ │ │ │ │ ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters_cropped.png
│ │ │ │ ├── ...
│ │ │ │ ├── frame_0001/
│ │ │ │ │ ├── skeletons/
│ │ │ │ │ │ ├── ...
│ │ │ │ │ ├── ...
│ │ │ ├── ...
│ │ ├── mutant/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│ │ │ │ ├── frame_0000/
│ │ │ │ │ ├── skeletons/
│ │ │ │ │ │ ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_skeleton_cell_1.png
│ │ │ │ │ │ ├── ...
│ │ │ │ │ ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_preprocessing.png
│ │ │ │ │ ├── ...
│ │ │ │ ├── frame_0001/
│ │ │ │ │ ├── skeletons/
│ │ │ │ │ │ ├── ...
│ │ │ │ │ ├── ...
│ │ │ ├── ...
│ │ ├── percentile_analysis/
│ │ │ ├── percentile_results.csv
│ │ │ ├── percentile_vs_num_cells.png
│ │ │ ├── percentile_comparison_90_95.png
│ │ ├── timing_analysis/
│ │ │ ├── speedup_comparison.png
│ │ │ ├── ...
│ ├── metrics/
│ │ ├── wildtype/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1_metrics.csv
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1_timings.csv
│ │ │ ├── ...
│ │ ├── mutant/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1_metrics.csv
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1_timings.csv
│ │ │ ├── ...
│ ├── morph_plots/
│ │ ├── correlation_heatmap.png
│ │ ├── histogram_plots.png
│ │ ├── ...
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
Processes all videos in `data/raw/wildtype/` and `data/raw/mutant/`, applying background correction, Fourier filtering, edge detection, and contour-based segmentation. Generates metrics (cell area, intensity, eccentricity, dendrite count) and visualizations (edge filters, contours, segmentation masks, histograms, skeleton overlays) for each frame. Optimized with parallel processing (using 4-8 processes) and configurable frame subsampling (default: process all frames) for efficient execution.

**Usage**:
```bash
python scripts/main_workflow.py
```
- Prompts for the number of frames to process (e.g., "10" or "all").
- Alternatively, specify parameters via command-line:
  ```bash
  python scripts/main_workflow.py --max-frames 10 --min-area 50 --use-percentile --percentile 90 --crop 40 150 250 512 --num-wildtype 3 --num-mutant 3 --subsample-rate 1
  ```
- `--max-frames`: Number of frames to process per video (default: None, prompts user).
- `--min-area`: Minimum area for segmented cells (default: 10).
- `--use-percentile`: Use percentile thresholding instead of Otsu (default: False).
- `--percentile`: Percentile for thresholding (default: 90, recommended based on analysis).
- `--crop`: Crop region as `y1 y2 x1 x2` (optional).
- `--num-wildtype`/`--num-mutant`: Number of videos to process per condition (default: None, all).
- `--subsample-rate`: Process every nth frame (default: 1, process all frames; e.g., 5 for every 5th frame).

**Outputs**:
- Processed images: `data/processed/<condition>/<video_name>/frame_XXXX/`
- Visualizations: `results/figures/<condition>/<video_name>/frame_XXXX/` (includes `skeletons/` subfolder for overlays comparing full skeletons vs dendrite skeletons used for counting).
- Metrics: `results/metrics/<condition>/<video_name>_metrics.csv`
- Records **per-frame timings** for preprocessing, segmentation, metrics, I/O, visualization, and total runtime. Saves timings to `results/metrics/<condition>/<video_name>_timings.csv`.

### analyze_percentiles.py
Analyzes cell segmentation across percentiles (80–99) to determine the optimal percentile for thresholding. Generates:
- A CSV file with cell counts per percentile (`percentile_results.csv`).
- A plot of cell counts versus percentile with the optimal percentile (elbow around 90) highlighted (`percentile_vs_num_cells.png`).
- A side-by-side comparison figure for percentiles 90–95, showing labeled cell images (with cell IDs) and the original (filtered) image (`percentile_comparison_90_95.png`).

**Usage**:
```bash
python scripts/analyze_percentiles.py
```
- Processes `data/raw/mutant/Confluence_Single movie_30.03.2025_no mask_G5_1.mp4` by default.
- Outputs saved in `results/figures/percentile_analysis/`.

**Choosing the Optimal Percentile**:
- Inspect `percentile_comparison_90_95.png` to compare labeled cells (top row) against the original image (bottom row) for percentiles 90–95.
- Select the percentile with clear cell boundaries, minimal noise (small spurious regions), and complete cell detection.
- Update `main_workflow.py` to set `--percentile X` (e.g., 91) if a better value is identified.

## Usage
Run the full pipeline:
```bash
python scripts/main_workflow.py
```
For specific parameters, including subsampling:
```bash
python scripts/main_workflow.py --max-frames 10 --min-area 50 --use-percentile --percentile 90 --subsample-rate 1
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
- `percentile_results.csv`: Cell counts per percentile.
- `percentile_vs_num_cells.png`: Plot of cell counts with optimal percentile.
- `percentile_comparison_90_95.png`: Side-by-side comparison of labeled cells and original image for percentiles 90–95.
- Timing analysis: `results/figures/timing_analysis/`
- `avg_time_per_stage.png`: Bar plot showing the average time for each processing stage.
- `speedup_comparison.png`: Bar plot comparing sequential and parallel processing times, showing average 8.0x speedup.
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
- Adjust `min_area` in `segment_cells` (in `src/segmentation.py`) if too few cells are detected (e.g., `min_area=5`).
- The default percentile in `main_workflow.py` is 94, based on initial analysis. Run `analyze_percentiles.py` to confirm or select a better value.
- dendrite_count is computed from protrusion skeletons: if a skeleton has fewer than two branches, it is considered non-dendritic (0).
- For performance with many frames, set a smaller `max_frames` (e.g., 10) or increase `--subsample-rate` (e.g., 5 for every 5th frame). Parallel processing (4-8 processes) improves runtime efficiency.

## Performance and Optimization
The osteocyte video analysis pipeline leverages parallel processing with 4-8 processes, achieving an **average 8.0x speedup** (e.g., 59.05 min to 7.38 min for `Confluence_Single movie_30.03.2025_no mask_C2_3` wildtype), as shown in `results/figures/timing_analysis/speedup_comparison.png`. Total CPU time for 6 videos (144 frames each) is **~350.38 min**, reduced to **~43.8 min** wall-clock time with 8 processes. Bar plot analysis (`timing_analysis.ipynb`) identifies Metrics (**45.61 s** wildtype, **43.34 s** mutant), Visualize (**23.81 s** wildtype, **23.52 s** mutant), and Crop (**4.62 s** wildtype, **4.21 s** mutant) as primary bottlenecks. Box plot results show high variability in per-frame processing times, particularly for C2_3 (wildtype, median **75.96 s**) and G5_1 (mutant, median **70.55 s**), with wildtype videos slightly slower, likely due to cell morphology complexity.

**Future Improvements**:
1. Optimize Metrics and Visualize stages through algorithmic improvements.
2. Enhance Crop stage with adaptive cropping based on frame content.
3. Leverage AWS high-performance computing (e.g., EC2 instances) for GPU-accelerated processing.
4. Reduce variability (e.g., in C2_3) with adaptive subsampling or dynamic resource allocation.
5. Implement asynchronous I/O with `aiofiles` to minimize overhead.
6. Enable parallel video processing and caching of intermediate results.

## Quick Timing Summary (optional)
After a run, you can quickly summarize average stage times in terminal (Python one-liner):
```bash
python - <<'PY'
import pandas as pd, glob
paths = glob.glob('results/metrics/*/*_timings.csv')
df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
cols = ['t_preprocess_s','t_segment_s','t_metrics_s','t_io_s','t_visualize_s','t_crop_total_s','t_total_s']
print(df.groupby('condition')[cols].mean().round(3))
PY
```