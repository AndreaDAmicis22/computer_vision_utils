import cv2
import numpy as np

from ..vision.utils import ensure_color


def draw_anomaly_map(img: np.ndarray, amap: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Draw anomaly map overlayed on image."""
    if normalize:
        a_min, a_max = amap.min(), amap.max()
        amap = (amap - a_min) / (a_max - a_min)
    heatmap = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)


def draw_segmentation(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Draw segmentation mask overlayed on image."""
    color = ensure_color(img)
    drawn = color.copy()
    drawn[mask > 0] = (255, 0, 0)
    return cv2.addWeighted(color, 1 - alpha, drawn, alpha, 0)
