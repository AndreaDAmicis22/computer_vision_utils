import cv2
import numpy as np


def normalize_brightness(img: np.ndarray, ksize: tuple[int, int] = (201, 201), mean: float = 90.0) -> np.ndarray:
    """Ensure image has homogeneous lightning.

    Args:
        img (numpy.ndarray): Image to correct in RGB format, shape `(h, w, 3)`.
        ksize (tuple[int, int]): Kernel size used to locally estimate brightness, default `(201, 201)`.
        mean (float): Mean brightness used for the normalization, default `90.0`.

    Returns:
        out (numpy.ndarray): Image with its brightness corrected, shape `(h, w, 3)`.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = img_hsv[..., 2].astype(np.float32)

    illumination = cv2.GaussianBlur(v, ksize, 0)
    illumination += 1e-6

    v_eq = v / illumination * mean
    v_eq = np.clip(v_eq, 0, 255).astype(np.uint8)

    img_hsv[..., 2] = v_eq
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
