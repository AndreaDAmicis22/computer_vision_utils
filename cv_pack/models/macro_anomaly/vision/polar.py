import cv2
import numpy as np
import polanalyser as pa

from ..vision.utils import normalize_img


def preprocess_polar(img: np.ndarray, size: tuple[int, int] = (1024, 1224)) -> np.ndarray:
    """Convert raw polar image to intensity."""
    img_000, img_045, img_090, img_135 = pa.demosaicing(img, pa.COLOR_PolarMono)
    img_list = [img_000, img_045, img_090, img_135]

    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(img_list, angles)

    intensity = pa.cvtStokesToIntensity(img_stokes)
    intensity = cv2.resize(intensity, size[::-1], cv2.INTER_NEAREST)

    return normalize_img(intensity)
