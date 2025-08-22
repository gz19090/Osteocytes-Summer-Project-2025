# image_utils.py

"""
image_utils.py
--------------
This file provides functions for preprocessing osteocyte cell video frames in the osteocyte culture project.
It handles video loading, frame conversion to grayscale, noise reduction, background correction, and image saving.
The preprocessing steps ensure consistent image quality for downstream segmentation and analysis.
Key functions include:
- load_video: Loads video frames, applies grayscale conversion, smoothing, cropping, and resizing.
- correct_background: Corrects uneven lighting using Gaussian blur and intensity rescaling.
- apply_fourier_filter: Reduces noise with a parabolic Fourier transform filter.
- save_image: Saves processed images as TIFF files.
Used in the main_workflow.py to prepare wildtype and mutant osteocyte videos for accurate cell segmentation and morphological analysis.
"""

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
import imageio.v2 as imageio

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video(video_path: str, crop_height: int = 860, resize_shape: Tuple[int, int] = (512, 512)) -> List[np.ndarray]:
    """Load and preprocess video frames for analysis.
    This function reads a video file, converts each frame to grayscale, smooths it to reduce noise,
    crops the frame to remove unwanted areas (like artifacts), and resizes it to a standard size.
    
    Args:
        video_path (str): Path to the video file.
        crop_height (int): Height to crop frames to (default: 860 pixels).
        resize_shape (Tuple[int, int]): Target size for frames (default: (512, 512)).
    
    Returns:
        List[np.ndarray]: List of processed frames (each a 2D grayscale array).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {video_path} not found.')
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise ValueError(f'No frames in {video_path}.')
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # Stop if no more frames
        
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Smooth the frame to reduce noise, keeping small details like dendrites
        frame = gaussian(frame, sigma=1.0)  # Sigma=1.0 preserves dendritic features
        
        # Crop the frame to remove unwanted areas (e.g., text or artifacts)
        frame = frame[:crop_height, :]
        if frame.size == 0:
            continue  # Skip empty frames
        
        # Resize the frame to a standard size for consistent analysis
        frame = resize(frame, resize_shape, mode='constant', anti_aliasing=True)
        frames.append(frame)
    
    # Release the video file
    cap.release()
    
    # Check if any frames were processed
    if not frames:
        raise ValueError('No frames processed.')
    
    return frames

def correct_background(image: np.ndarray) -> np.ndarray:
    """Correct uneven lighting in an image.
    This function estimates the background using a strong blur, divides the image by the background
    to even out lighting, and rescales the result to a 0–1 range for consistent intensity.
    
    Args:
        image (np.ndarray): 2D grayscale image to correct.
    
    Returns:
        np.ndarray: Corrected image with even lighting (values in [0, 1]).
    """
    # Estimate the background with a strong blur to smooth out details
    background = gaussian(image, sigma=20, preserve_range=True)  # Sigma=20 avoids losing small features
    
    # Divide the image by the background (adjusted to its mean) to correct lighting
    corrected = image / (background / np.mean(background))
    
    # Rescale the image to a 0–1 range for consistent intensity
    return rescale_intensity(corrected, out_range=(0, 1))

def apply_fourier_filter(image: np.ndarray) -> np.ndarray:
    """Apply a Fourier filter to reduce noise in an image.
    This function uses a Fourier transform to filter out certain frequencies, keeping important
    features like cell boundaries while reducing noise, and rescales the result to 0–1.
    
    Args:
        image (np.ndarray): 2D grayscale image to filter.
    
    Returns:
        np.ndarray: Filtered image (values in [0, 1]).
    """
    # Center the image by subtracting its mean
    img = image.astype(float) - image.mean()
    
    # Compute the 2D Fourier transform
    fft = np.fft.fft2(img)
    
    # Shift the zero frequency to the center
    fft_shifted = np.fft.fftshift(fft)
    
    # Create a parabolic mask to filter specific frequencies
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    center_row = rows // 2
    parabola_width, parabola_height = 150, 30
    col_indices = np.arange(cols) - cols // 2
    top_parabola = center_row + parabola_height * (col_indices / parabola_width) ** 2
    bottom_parabola = center_row - parabola_height * (col_indices / parabola_width) ** 2
    r, c = np.indices((rows, cols))
    for c in range(cols):
        mask[:, c] = (r[:, c] >= bottom_parabola[c]) & (r[:, c] <= top_parabola[c])
    
    # Smooth the mask to avoid sharp edges
    mask = gaussian(mask, sigma=100)
    
    # Apply the mask to the Fourier transform
    masked_fft = fft_shifted * mask
    
    # Reverse the shift and transform back to an image
    product = np.fft.ifftshift(masked_fft)
    recovered = np.fft.ifft2(product).real
    
    # Rescale the image to a 0–1 range
    recovered = rescale_intensity(recovered, out_range=(0, 1))
    
    # Handle any invalid values (NaN or Inf) from the transform
    if np.any(np.isnan(recovered)) or np.any(np.isinf(recovered)):
        logger.warning("Fourier filter output contains NaN/Inf. Replacing with zeros.")
        recovered = np.nan_to_num(recovered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Log the output details for debugging
    logger.debug(f"Fourier filter output shape: {recovered.shape}, min: {recovered.min()}, max: {recovered.max()}")
    
    return recovered

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to a file.
    This function saves a 2D or 3D image as a TIFF file, creating the output folder if needed.
    
    Args:
        image (np.ndarray): Image to save (2D or 3D array).
        output_path (str): File path to save the image (e.g., TIFF or PNG).
    """
    try:
        # Create the output folder if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image as a TIFF file
        imageio.imwrite(output_path, image, format='TIFF')
        logger.info(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        raise