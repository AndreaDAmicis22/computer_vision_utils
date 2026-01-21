import random

import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    """
    Aggiunge rumore gaussiano all'immagine.
    """
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def equalize_histogram(img_gray):
    """
    Equalizzazione dell'istogramma per immagini grayscale.
    """
    return cv2.equalizeHist(img_gray)


def apply_clahe(img_bgr: np.ndarray, clip_limit=4, tile_grid_size=(64, 64)) -> np.ndarray:
    """
    Applica CLAHE sul canale L dell'immagine in LAB.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def affine_transform(img, matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """
    Applica trasformazione affine usando una matrice 2x3.
    """
    return cv2.warpAffine(img, matrix, output_shape)


def edge_sobel(img, dx=1, dy=1, ksize=3):
    return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)


def edge_canny(img, threshold1=100, threshold2=200):
    return cv2.Canny(img, threshold1, threshold2)


def increase_contrast(img_bgr: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Aumenta il contrasto dell'immagine.
    factor > 1 aumenta il contrasto, factor < 1 lo diminuisce.
    """
    img_float = img_bgr.astype(np.float32) / 255.0
    mean = img_float.mean()
    img_contrast = np.clip((img_float - mean) * factor + mean, 0, 1)
    return (img_contrast * 255).astype(np.uint8)


def increase_saturation(img_bgr: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Aumenta la saturazione dell'immagine.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(float) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gamma_correction(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return (img * 255).astype(np.uint8)


def apply_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applica un filtro mediano per rimuovere rumore impulsivo.

    Args:
        img (np.ndarray): Immagine input.
        ksize (int): Dimensione del kernel (deve essere dispari).

    Returns:
        np.ndarray: Immagine filtrata.
    """
    if ksize % 2 == 0:
        ksize += 1  # assicuriamoci che sia dispari
    return cv2.medianBlur(img, ksize)


def apply_gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applica un filtro gaussiano per smoothing.

    Args:
        img (np.ndarray): Immagine input.
        ksize (int): Dimensione del kernel (deve essere dispari).
        sigma (float): Deviazione standard del filtro gaussiano.

    Returns:
        np.ndarray: Immagine filtrata.
    """
    if ksize % 2 == 0:
        ksize += 1  # assicuriamoci che sia dispari
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def fuse_and_enhance(
    self,
    imgs_bgr: list[np.ndarray],
    clahe_clip: float = 4.0,
    clahe_tile: tuple[int, int] = (64, 64),
    sat_factor: float = 1.1,
    contrast_factor: float = 1.1,
) -> np.ndarray:
    """
    Fonde più immagini BGR e applica enhancement:
    - CLAHE
    - aumento saturazione
    - aumento contrasto
    """
    # Fonde prendendo il massimo pixel-wise
    fused = np.max(np.stack(imgs_bgr, axis=0), axis=0)
    fused = self._apply_clahe(fused, clahe_clip, clahe_tile)
    fused = self._increase_saturation(fused, sat_factor)
    return self._increase_contrast(fused, contrast_factor)


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def flip_vertical(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Ruota l'immagine intorno al centro.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def morphological_open(img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def morphological_close(img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalizza l'immagine a [0,1].
    """
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def resize(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Ridimensiona immagine mantenendo il canale.
    """
    return cv2.resize(img, target_shape[::-1], interpolation=cv2.INTER_LINEAR)


def random_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """
    Esegue un crop casuale.
    """
    h, w = img.shape[:2]
    ch, cw = crop_size
    top = random.randint(0, max(0, h - ch))
    left = random.randint(0, max(0, w - cw))
    return img[top : top + ch, left : left + cw]


def sharpen(img, amount=1.2, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def warp_perspective(img, src_pts, dst_pts, output_shape: tuple[int, int]):
    """
    Applica trasformazione prospettica.
    src_pts e dst_pts devono essere array 4x2.
    """
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(img, M, output_shape)


def zoom_in(img: np.ndarray, zoom_factor: float = 1.02) -> np.ndarray:
    """
    Applica uno zoom-in centrato sull'immagine.
    zoom_factor >1 zoom-in, <1 zoom-out
    """
    h, w = img.shape[:2]

    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    cropped = img[top : top + new_h, left : left + new_w]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
