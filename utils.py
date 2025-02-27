import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Forzar el backend TkAgg
import matplotlib.pyplot as plt

def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def visualize_prediction(image, mask=None, prediction=None, cropped_image=None, filename="output.png"):
    num_images = 2 + (mask is not None) + (cropped_image is not None)
    fig, ax = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

    # Verificar formato de la imagen antes de transponer
    if image.shape[0] in [1, 3]:  # Caso [C, H, W]
        image = image.transpose(1, 2, 0)

    ax[0].imshow(image)
    ax[0].set_title("Imagen")
    ax[0].axis("off")

    index = 1
    if mask is not None:
        ax[index].imshow(mask.squeeze(), cmap="gray")
        ax[index].set_title("Máscara Real")
        ax[index].axis("off")
        index += 1

    ax[index].imshow(prediction.squeeze(), cmap="gray")
    ax[index].set_title("Predicción")
    ax[index].axis("off")
    index += 1

    if cropped_image is not None:
        ax[index].imshow(cropped_image, cmap="gray")
        ax[index].set_title("Imagen Recortada")
        ax[index].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
