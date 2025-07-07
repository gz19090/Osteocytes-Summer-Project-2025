# Osteocyte 2D Cell Culture Analysis

This project analyzes videos of 2D osteocyte cell cultures to segment and quantify cells and dendritic processes using parabolic FFT filtering and contour-based segmentation. It processes videos in `wildtype` and `mutant` conditions, generating metrics (cell area, intensity, eccentricity, dendritic length) and visualizations (edge filter plots, contour overlays, segmentation masks, histograms) for user-specified frames.

## Project Structure

```
Osteocytes culture/
├── src/
│   ├── __init__.py
│   ├── image_utils.py        # Functions for video loading, background correction, Fourier filtering, image saving
│   ├── segmentation.py       # Edge filter optimization and contour-based segmentation
│   ├── analysis.py           # Cell and dendritic process analysis
│   ├── visualization.py      # Plotting functions for edge filters, contours, segmentation, histograms
├── scripts/
│   ├── main_workflow.py      # Main script to process all videos and frames
├── notebooks/
│   ├── 01_preprocessing.ipynb  # Background correction, Fourier filtering, edge optimization
│   ├── 02_segmentation.ipynb   # Contour-based cell segmentation
│   ├── 03_analysis.ipynb       # Full analysis and visualization
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
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_filtered.tif
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.tif
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled.tif
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_cropped.tif
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined_cropped.tif
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_labeled_cropped.tif
│   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│   │   │   │   ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_filtered.tif
│   │   │   │   ├── ...
│   │   │   ├── ...
├── results/
│   ├── figures/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1/
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_preprocessing.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_combined.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_contours.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_segmentation.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_histograms.png
│   │   │   │   ├── wildtype_Confluence_Single movie_30.03.2025_no mask_C4_1_frame_0000_edge_filters_cropped.png
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1/
│   │   │   │   ├── mutant_Confluence_Single movie_30.03.2025_no mask_G5_1_frame_0000_preprocessing.png
│   │   │   │   ├── ...
│   │   │   ├── ...
│   ├── metrics/
│   │   ├── wildtype/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_C4_1_metrics.csv
│   │   │   ├── ...
│   │   ├── mutant/
│   │   │   ├── Confluence_Single movie_30.03.2025_no mask_G5_1_metrics.csv
│   │   │   ├── ...
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

## Usage
Run the full pipeline to process all videos:
```bash
python scripts/main_workflow.py
```
- When prompted, enter the number of frames to process (e.g., "10" for the first 10 frames, or "all" for all frames).
- Alternatively, specify the frame limit via command-line:
  ```bash
  python scripts/main_workflow.py --max-frames 10
  ```

Explore interactively using Jupyter notebooks:
- `notebooks/01_preprocessing.ipynb`: Background correction, Fourier filtering, edge filter optimization. Outputs saved in `data/processed/{condition}/{video_name}/` and `results/figures/{condition}/{video_name}/`.
- `notebooks/02_segmentation.ipynb`: Contour-based cell segmentation. Outputs saved in `data/processed/{condition}/{video_name}/` and `results/figures/{condition}/{video_name}/`.
- `notebooks/03_analysis.ipynb`: Full analysis (cell metrics, dendritic length) and visualizations. Metrics saved in `results/metrics/{condition}/{video_name}_metrics.csv`.

To run a notebook:
```bash
cd /.../Osteocytes-Summer-Project-2025/Osteocytes culture
jupyter notebook notebooks/01_preprocessing.ipynb
```
- Enter the number of frames when prompted (e.g., "10" or "all").

## Outputs
- **Processed Images**: Saved in `data/processed/{condition}/{video_name}/` (e.g., `filtered.tif`, `combined.tif`, `labeled.tif`, `cropped.tif`, `combined_cropped.tif`, `labeled_cropped.tif`).
- **Visualizations**: Saved in `results/figures/{condition}/{video_name}/` (e.g., `preprocessing.png`, `edge_filters.png`, `combined.png`, `contours.png`, `segmentation.png`, `histograms.png`).
- **Metrics**: Saved in `results/metrics/{condition}/{video_name}_metrics.csv` with columns: `label`, `area`, `mean_intensity`, `eccentricity`, `dendritic_length`, `video`, `frame`, `condition`.

## Dependencies
See `requirements.txt`. Key libraries:
- `numpy`
- `scikit-image`
- `opencv-python`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `cellpose` (optional, commented out due to MPS issues on M4 Mac)

## Notes
- Videos must be in MP4 format and placed in `data/raw/wildtype/` or `data/raw/mutant/`.
- Adjust `min_area` in `segment_cells` (in `src/segmentation.py`) if too few cells are detected (e.g., `min_area=5`).
- Use percentile thresholding (`use_percentile=True` in `segment_cells`) instead of Otsu if needed.
- For performance with many frames, set a smaller `max_frames` (e.g., 10).
- Cellpose is disabled due to MPS compatibility issues on M4 Mac. Uncomment relevant sections to enable if compatible hardware is available.