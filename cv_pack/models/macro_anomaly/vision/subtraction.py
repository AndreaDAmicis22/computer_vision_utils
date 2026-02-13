import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .utils import float_to_int, ensure_grayscale


def background_subtraction(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Perform background subtraction."""


def image_difference_ssim(img: np.ndarray, ref: np.ndarray, win_size: int = 41) -> np.ndarray:
    """Perform image difference using structural similarity."""
    img_gray = ensure_grayscale(img)
    ref_gray = ensure_grayscale(ref)
    diff = 1 - (ssim(ref_gray, img_gray, win_size=win_size, full=True)[1] + 1) / 2
    return float_to_int(diff)


def image_difference_mse(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Perform image difference using mean squared error (mean is across channels)."""
    diff = cv2.absdiff(img, ref)
    if diff.ndim == 3:
        diff = np.mean(diff, axis=-1)
        diff = np.round(diff).astype(np.uint8)
    return diff
