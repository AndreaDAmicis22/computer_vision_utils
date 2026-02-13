import cv2
import numpy as np
import torch


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image in range [0, 255]."""
    i_max, i_min = img.max(), img.min()
    img = (img - i_min) / (i_max - i_min)
    return (255 * img).astype(np.uint8)


def float_to_int(img: np.ndarray) -> np.ndarray:
    """Convert float image in [0.0, 1.0] to int image in range [0, 255]."""
    return (255 * img).astype(np.uint8)


def int_to_float(img: np.ndarray) -> np.ndarray:
    """Convert int image in range [0, 255] to float image in [0.0, 1.0]."""
    return (img / 255).astype(np.float32)


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure image is grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img


def ensure_color(img: np.ndarray) -> np.ndarray:
    """Ensure image is RGB."""
    if img.ndim == 3:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to tensor."""
    if img.dtype == np.uint8:
        img = img / 255
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    return img_tensor.permute(2, 0, 1)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor numpy image."""
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return img
