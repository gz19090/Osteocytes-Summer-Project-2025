# image_utils.py

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.util import invert
from skimage.filters import threshold_local
from pathlib import Path
from typing import List, Tuple
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video(video_path: str, crop_height: int = 860, resize_shape: Tuple[int, int] = (512, 512)) -> List[np.ndarray]:
    """Load video and preprocess frames.
    
    This function opens a video file, reads frames, converts them to grayscale, applies Gaussian smoothing
    to reduce noise, crops to a specified height (e.g., to remove artifacts), and resizes to a target shape
    with anti-aliasing. It handles empty videos or frames and raises errors for invalid files.

    Args:
        video_path (str): Path to video file.
        crop_height (int): Height to crop frames to (default: 860).
        resize_shape (Tuple[int, int]): Target frame size (default: (512, 512)).
    
    Returns:
        List[np.ndarray]: List of preprocessed frames (each a 2D grayscale array).
    """
    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {video_path} not found.')
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise ValueError(f'No frames in {video_path}.')
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian smoothing to reduce noise; sigma=1.0 for better preservation of dendritic features
        frame = gaussian(frame, sigma=1.0)  # Increased from 0.5 to better preserve details like dendrites
        # Crop to specified height to remove potential footer or header artifacts
        frame = frame[:crop_height, :]
        if frame.size == 0:
            continue
        # Resize with constant mode and anti-aliasing for consistent analysis dimensions
        frame = resize(frame, resize_shape, mode='constant', anti_aliasing=True)
        frames.append(frame)
    
    # Release the capture resource
    cap.release()
    if not frames:
        raise ValueError('No frames processed.')
    return frames

def correct_background(image: np.ndarray) -> np.ndarray:
    """Apply background correction.
    
    This function estimates the background using Gaussian blur and normalizes the image by dividing
    by the background mean. It then rescales intensity to [0, 1]. This helps correct uneven illumination
    common in microscopy images. The sigma=20 balances smoothing without over-blurring small features.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
    
    Returns:
        np.ndarray: Background-corrected image (same shape, values in [0, 1]).
    """
    # Estimate background with Gaussian blur; sigma=20 to avoid over-smoothing small features like dendrites
    background = gaussian(image, sigma=20, preserve_range=True)  # Reduced from 50 for better detail preservation
    # Normalize by dividing by background adjusted to its mean
    corrected = image / (background / np.mean(background))
    # Rescale to [0, 1] for consistent intensity range
    return rescale_intensity(corrected, out_range=(0, 1))

# Optional alternative with local thresholding (commented out; use if needed for adaptive refinement)
# def correct_background(image: np.ndarray) -> np.ndarray:
#     """Apply background correction with adaptive local thresholding.
#     
#     Args:
#         image (np.ndarray): Input image.
#     
#     Returns:
#         np.ndarray: Background-corrected image.
#     """
#     background = gaussian(image, sigma=20, preserve_range=True)
#     corrected = image / (background / np.mean(background))
#     corrected = rescale_intensity(corrected, out_range=(0, 1))
#     # Add adaptive local thresholding to refine (e.g., for variable lighting)
#     local_thresh = threshold_local(corrected, block_size=35, method='gaussian')
#     corrected = corrected > local_thresh
#     return corrected

def apply_fourier_filter(image: np.ndarray) -> np.ndarray:
    """Apply parabolic Fourier transform filter.
    
    This function normalizes the image, applies a 2D FFT, shifts it, multiplies by a parabolic mask to
    filter frequencies, inverse shifts and transforms, and rescales the result. It enhances features like
    cell edges and dendrites. Logging checks for NaNs/Inf, which can occur in FFT operations.

    Args:
        image (np.ndarray): Input image (2D grayscale array).
    
    Returns:
        np.ndarray: Filtered image (same shape, values in [0, 1]).
    """
    # Subtract mean for zero-centering before FFT
    img = image.astype(float) - image.mean()
    # Compute 2D FFT
    fft = np.fft.fft2(img)
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    rows, cols = img.shape
    # Create parabolic mask for frequency filtering
    mask = np.zeros((rows, cols))
    center_row = rows // 2
    parabola_width, parabola_height = 150, 30
    col_indices = np.arange(cols) - cols // 2
    top_parabola = center_row + parabola_height * (col_indices / parabola_width) ** 2
    bottom_parabola = center_row - parabola_height * (col_indices / parabola_width) ** 2
    r, c = np.indices((rows, cols))
    for c in range(cols):
        mask[:, c] = (r[:, c] >= bottom_parabola[c]) & (r[:, c] <= top_parabola[c])
    # Smooth the mask with Gaussian
    mask = gaussian(mask, sigma=100)
    
    # Apply mask to shifted FFT
    masked_fft = fft_shifted * mask
    # Inverse shift and FFT to recover image (real part only)
    product = np.fft.ifftshift(masked_fft)
    recovered = np.fft.ifft2(product).real
    # Rescale to [0, 1]
    recovered = rescale_intensity(recovered, out_range=(0, 1))
    
    # Check for NaNs or Inf values (can occur in FFT if input has extremes)
    if np.any(np.isnan(recovered)) or np.any(np.isinf(recovered)):
        logger.warning("Fourier filter output contains NaN/Inf. Replacing with zeros.")
        recovered = np.nan_to_num(recovered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Debug log for output shape and range
    logger.debug(f"Fourier filter output shape: {recovered.shape}, min: {recovered.min()}, max: {recovered.max()}")
    return recovered

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file.
    
    This function saves the image using skimage.io.imsave, disabling contrast check for efficiency.

    Args:
        image (np.ndarray): Image to save (2D or 3D array).
        output_path (str): Output file path (e.g., TIFF or PNG).
    """
    from skimage.io import imsave
    # Save the image without checking contrast for faster execution
    imsave(output_path, image, check_contrast=False)