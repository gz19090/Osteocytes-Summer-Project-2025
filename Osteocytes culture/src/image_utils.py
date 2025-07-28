import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.util import invert
from pathlib import Path
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video(video_path: str, crop_height: int = 860, resize_shape: Tuple[int, int] = (512, 512)) -> List[np.ndarray]:
    """Load video and preprocess frames.
    
    Args:
        video_path (str): Path to video file.
        crop_height (int): Height to crop frames to.
        resize_shape (Tuple[int, int]): Target frame size.
    
    Returns:
        List[np.ndarray]: List of preprocessed frames.
    """
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = gaussian(frame, sigma=0.5)
        frame = gaussian(frame, sigma=1.0) # try increased sigma to better preserve dendritic features
        frame = frame[:crop_height, :]
        if frame.size == 0:
            continue
        frame = resize(frame, resize_shape, mode='constant', anti_aliasing=True)
        frames.append(frame)
    
    cap.release()
    if not frames:
        raise ValueError('No frames processed.')
    return frames

def correct_background(image: np.ndarray) -> np.ndarray:
    """Apply background correction.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Background-corrected image.
    """
    # background = gaussian(image, sigma=50, preserve_range=True)
    background = gaussian(image, sigma=20, preserve_range=True) # avoid over-smoothing small features
    corrected = image / (background / np.mean(background))
    return rescale_intensity(corrected, out_range=(0, 1))

# def apply_fourier_filter(image: np.ndarray) -> np.ndarray:
#     img = image.astype(float) - image.mean()
#     fft = np.fft.fft2(img)
#     fft_shifted = np.fft.fftshift(fft)
#     rows, cols = img.shape
#     mask = np.zeros((rows, cols))
#     center_row = rows // 2
#     parabola_width, parabola_height = 150, 30
#     col_indices = np.arange(cols) - cols // 2
#     top_parabola = center_row + parabola_height * (col_indices / parabola_width) ** 2
#     bottom_parabola = center_row - parabola_height * (col_indices / parabola_width) ** 2
#     r, c = np.indices((rows, cols))
#     for c in range(cols):
#         mask[:, c] = (r[:, c] >= bottom_parabola[c]) & (r[:, c] <= top_parabola[c])
#     mask = gaussian(mask, sigma=100)
#     masked_fft = fft_shifted * mask
#     product = np.fft.ifftshift(masked_fft)
#     recovered = np.fft.ifft2(product).real
#     recovered = rescale_intensity(recovered, out_range=(0, 1))
#     logger.debug(f"Fourier filter output shape: {recovered.shape}, has NaN: {np.any(np.isnan(recovered))}")
#     return recovered

def apply_fourier_filter(image: np.ndarray) -> np.ndarray:
    import logging
    logger = logging.getLogger(__name__)
    img = image.astype(float) - image.mean()
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
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
    mask = gaussian(mask, sigma=100)
    masked_fft = fft_shifted * mask
    product = np.fft.ifftshift(masked_fft)
    recovered = np.fft.ifft2(product).real
    recovered = rescale_intensity(recovered, out_range=(0, 1))
    # Check for Nans and extreme values
    if np.any(np.isnan(recovered)) or np.any(np.isinf(recovered)):
        logger.warning("Fourier filter output contains NaN/Inf. Replacing with zeros.")
        recovered = np.nan_to_num(recovered, nan=0.0, posinf=0.0, neginf=0.0)
    logger.debug(f"Fourier filter output shape: {recovered.shape}, min: {recovered.min()}, max: {recovered.max()}")
    return recovered

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file.
    
    Args:
        image (np.ndarray): Image to save.
        output_path (str): Output file path.
    """
    from skimage.io import imsave
    imsave(output_path, image, check_contrast=False)