import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask_green(image, mask, alpha=0.4):
    """
    Sovrappone una maschera verde su un'immagine.

    Args:
        image (np.ndarray): Immagine RGB o grayscale.
        mask (np.ndarray): Maschera binaria (0/1 o 0/255).
        alpha (float): Trasparenza della maschera.

    Returns:
        overlay (np.ndarray): Immagine RGB con overlay verde.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()

    mask_bin = (mask > 0).astype(np.uint8)
    overlay = image_rgb.copy()

    green = np.zeros_like(image_rgb)
    green[..., 1] = 255  # canale verde

    overlay[mask_bin == 1] = cv2.addWeighted(image_rgb[mask_bin == 1], 1 - alpha, green[mask_bin == 1], alpha, 0)

    return overlay


def plot_images_grid(images, titles=None, cols=3, figsize=(12, 6)):
    """
    Plotta una griglia di immagini.

    Args:
        images (list[np.ndarray]): Lista di immagini.
        titles (list[str], optional): Titoli per ciascuna immagine.
        cols (int): Numero di colonne.
        figsize (tuple): Dimensione figura matplotlib.
    """
    n_images = len(images)
    rows = int(np.ceil(n_images / cols))

    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)

        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        if titles is not None:
            ax.set_title(titles[i])

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_n_examples(folder_path, N=5, color_space="rgb", figsize=(15, 5)):
    """
    Plot casuale di N immagini da una cartella.

    Args:
        folder_path (str): Path alla cartella contenente immagini.
        N (int): Numero di immagini da plottare.
        color_space (str): 'rgb' o 'grayscale'.
        figsize (tuple): Dimensione figura matplotlib.
    """
    from . import random_image as ri

    images = []
    titles = []

    for _ in range(N):
        img, filename = ri.load_random_image(folder_path, color_space=color_space)
        images.append(img)
        titles.append(filename)

    plot_images_grid(images, titles=titles, cols=N, figsize=figsize)


def plot_grayscale_histogram(image) -> None:
    """Plot grayscale histogram of an image."""

    if image is None:
        msg = "Image not found or unreadable"
        raise ValueError(msg)

    hist: np.ndarray = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Count")
    plt.title("Grayscale Histogram")
    plt.show()


def plot_histogram(image, bins=256, title="Histogram", figsize=(8, 4)):
    """
    Plotta l'istogramma dei valori dei pixel.

    Args:
        image (np.ndarray): Immagine grayscale o RGB.
        bins (int): Numero di bin.
        title (str): Titolo del plot.
    """
    plt.figure(figsize=figsize)

    if len(image.shape) == 2:  # grayscale
        plt.hist(image.ravel(), bins=bins, color="black")
    else:
        colors = ("r", "g", "b")
        for i, col in enumerate(colors):
            plt.hist(image[..., i].ravel(), bins=bins, color=col, alpha=0.5)

    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()


def plot_image_comparison(image1, image2, title1="Original", title2="Processed", figsize=(10, 5)):
    """
    Confronto di due immagini affiancate.

    Args:
        image1, image2 (np.ndarray): Immagini da confrontare.
        title1, title2 (str): Titoli.
        figsize (tuple): Dimensione figura matplotlib.
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(image1 if len(image1.shape) == 3 else image1, cmap="gray")
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2 if len(image2.shape) == 3 else image2, cmap="gray")
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_mask_overlay_multiple(image, masks, alphas=None, colors=None, figsize=(6, 6)):
    """
    Sovrappone più maschere colorate su un'immagine.

    Args:
        image (np.ndarray): Immagine di base.
        masks (list[np.ndarray]): Lista di maschere binarie.
        alphas (list[float], optional): Trasparenze delle maschere.
        colors (list[tuple], optional): Colori RGB delle maschere.
    """
    overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if alphas is None:
        alphas = [0.4] * len(masks)
    if colors is None:
        default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        colors = default_colors * ((len(masks) + len(default_colors) - 1) // len(default_colors))

    for mask, alpha, color in zip(masks, alphas, colors, strict=False):
        mask_bin = (mask > 0).astype(np.uint8)
        color_layer = np.zeros_like(overlay)
        color_layer[..., 0] = color[0]
        color_layer[..., 1] = color[1]
        color_layer[..., 2] = color[2]

        overlay[mask_bin == 1] = cv2.addWeighted(
            overlay[mask_bin == 1], 1 - alpha, color_layer[mask_bin == 1], alpha, 0
        )

    plt.figure(figsize=figsize)
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
