from typing import Sequence

import cv2
import numpy as np


def align(segments: Sequence[np.ndarray] | np.ndarray, shift: int = 6) -> np.ndarray:
    """Align segments to reconstruct full image."""
    n = len(segments)
    aligned = []
    for idx, segment in enumerate(segments):
        aligned.append(cv2.copyMakeBorder(segment, 0, 0, (n - idx) * shift, idx * shift, cv2.BORDER_CONSTANT, value=0))
    return np.vstack(aligned)


def revert(img: np.ndarray, shift: int = 6, band: int = 1000) -> list[np.ndarray]:
    """Splits aligned image into its component segments."""
    h, w = img.shape
    n = h // band
    segments = []
    if img.ndim == 3:
        reshaped = img.reshape(n, band, w, 3)
    else:
        reshaped = img.reshape(n, band, w)
    for idx, segment in enumerate(reshaped):
        segments.append(segment[:, (n - idx) * shift : w - idx * shift])
    return segments
