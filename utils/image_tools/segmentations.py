import cv2
import numpy as np


def get_skeleton(mask):
    skeleton = np.zeros(mask.shape, np.uint8)
    img = mask.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def get_yolo_mask(yolo_results, original_shape, segmenter_mask):
    # create global yolo mask
    full_mask = np.zeros(original_shape[:2], dtype=np.uint8)
    for r in yolo_results:
        full_mask = cv2.bitwise_or(full_mask, r._mask)

    # delete detections outside the ellipse
    return cv2.bitwise_and(full_mask, segmenter_mask)


def segment_main_object(image_bgr: np.ndarray) -> np.ndarray:
    """Segment the main object using classical CV (illumination correction + watershed).

    Args:
        image_bgr: Input BGR image.

    Returns:
        Binary mask of the main object (uint8: 0 or 255).
    """
    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.threshold(image_bgr, 30, 255, cv2.THRESH_BINARY)[1]


def segment_by_histogram_peak(image: np.ndarray, min_intensity: int = 40) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Segment an image using the first significant histogram peak and convex hull.

    Pipeline:
    - Grayscale conversion
    - Histogram computation
    - First peak detection (intensity > min_intensity)
    - Threshold = peak + 10%
    - Binarization
    - Largest contour selection
    - Convex hull computation
    - Mask application

    Args:
        image: Input BGR or grayscale image.
        min_intensity: Minimum intensity to consider for peak detection.

    Returns:
        masked_image: Original image masked by the convex hull.
        mask: Binary mask of the convex hull.
        threshold: Threshold value used for binarization.
    """
    if image.ndim == 3:
        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Histogram
    hist: np.ndarray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # First peak detection
    peak_intensity: int = -1
    for i in range(max(min_intensity + 1, 1), 255):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peak_intensity = i
            break
    if peak_intensity < 0:
        msg = "No valid histogram peak found"
        raise RuntimeError(msg)
    # Threshold = peak + 5%
    threshold: int = int(peak_intensity * 1.05)
    threshold = min(threshold, 255)
    # Binarization
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        msg = "No contours found after thresholding"
        raise RuntimeError(msg)

    # Largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Convex hull
    hull: np.ndarray = cv2.convexHull(largest_contour)
    # Mask from convex hull
    mask: np.ndarray = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    # Apply mask
    masked_image: np.ndarray = cv2.bitwise_and(image, image, mask=mask)

    return masked_image, mask, threshold


def watershed_binarization(image_bgr: np.ndarray, min_distance_ratio: float = 0.14) -> tuple[np.ndarray, np.ndarray]:
    """Apply marker-controlled watershed for image binarization.

    Args:
        image_bgr: Input BGR image.
        min_distance_ratio: Threshold ratio for distance transform peak selection.

    Returns:
        Tuple containing:
        - binary_mask: Final binary mask (uint8, 0 or 255).
        - watershed_labels: Labeled watershed output.
    """
    # 1. Grayscale
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground is white
    if np.mean(gray[binary == 255]) < np.mean(gray[binary == 0]):
        binary = cv2.bitwise_not(binary)

    # 3. Morphological opening
    kernel: np.ndarray = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Distance transform (sure foreground)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, min_distance_ratio * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # 6. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marker labeling
    _num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 8. Watershed
    markers = cv2.watershed(image_bgr, markers)

    # 9. Binary output (exclude boundaries)
    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    binary_mask[markers > 1] = 255

    return binary_mask, markers
