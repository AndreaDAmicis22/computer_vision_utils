import os
import random

import cv2


def load_random_image(folder_path, color_space="rgb"):
    """
    Carica un'immagine casuale da una directory.

    Args:
        folder_path (str): Path alla directory contenente le immagini.
        color_space (str): "rgb" oppure "grayscale".

    Returns:
        img (np.ndarray): Immagine caricata.
        filename (str): Nome del file selezionato.
    """
    assert color_space in ["rgb", "grayscale"], "color_space deve essere 'rgb' o 'grayscale'"

    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

    if len(images) == 0:
        msg = f"Nessuna immagine trovata in {folder_path}"
        raise RuntimeError(msg)

    filename = random.choice(images)
    img_path = os.path.join(folder_path, filename)

    if color_space == "grayscale":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        msg = f"Errore nel caricamento dell'immagine: {img_path}"
        raise RuntimeError(msg)

    return img, filename
