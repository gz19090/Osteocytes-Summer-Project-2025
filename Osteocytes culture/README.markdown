# Osteocyte 2D Cell Culture Analysis
This project analyzes videos of 2D osteocyte cell cultures to segment and quantify cells and dendritic processes using parabolic FFT filtering and contour-based segmentation. It processes videos in `wildtype` and `mutant` conditions, generating metrics (cell area, intensity, eccentricity, dendrite count) and visualizations (edge filter plots, contour overlays, segmentation masks, histograms, skeleton overlays) for user-specified frames.

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
│ ├── metrics/
│ │ ├── wildtype/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_C4_1_metrics.csv
│ │ │ ├── ...
│ │ ├── mutant/
│ │ │ ├── Confluence_Single movie_30.03.2025_no mask_G5_1_metrics.csv
│ │ │ ├── ...
│ ├── morph_plots/
│ │ ├── correlation_heatmap.png
│ │ ├── correlation_network.png
│ │ ├── histogram_plots.png
│ │ ├── ...
├── README.md
├── requirements.txt
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/gz19090/Osteocytes-Summer-Project-2025.git
   cd Osteocytes-Summer-Project-2025/2d-cell-cultures/Osteocytes culture
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
- `--percentile`: Percentile for thresholding (default: 94, recommended based on analysis).
- `--crop`: Crop region as `y1 y2 x1 x2` (optional).
- `--num-wildtype`/`--num-mutant`: Number of videos to process per condition (default: None, all).
- `--subsample-rate`: Process every nth frame (default: 1, process all frames; e.g., 5 for every 5th frame).

**Outputs**:
- Processed images: `data/processed/<condition>/<video_name>/frame_XXXX/`
- Visualizations: `results/figures/<condition>/<video_name>/frame_XXXX/` (includes `skeletons/` subfolder for overlays comparing full skeletons vs dendrite skeletons used for counting).
- Metrics: `results/metrics/<condition>/<video_name>_metrics.csv`

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

## Outputs
- Processed images and visualizations: `data/processed/` and `results/figures/`
- Metrics: `results/metrics/`
- Morphological plots: `results/morph_plots/`
- Percentile analysis: `results/figures/percentile_analysis/`
- `percentile_results.csv`: Cell counts per percentile.
- `percentile_vs_num_cells.png`: Plot of cell counts with optimal percentile.
- `percentile_comparison_90_95.png`: Side-by-side comparison of labeled cells and original image for percentiles 90–95.

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