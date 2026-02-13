import cv2
import numpy as np


def segment_anomaly(amap: np.ndarray, thresh: float = 0.3, min_area: float = 2000.0) -> np.ndarray:
    """Segment anomaly map with specified threshold."""
    mask = (amap > thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    anomaly_map = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > min_area:
            anomaly_map[labels == label] = 255

    return anomaly_map
