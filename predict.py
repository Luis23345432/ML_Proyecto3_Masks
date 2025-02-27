import torch
import numpy as np
import os
import cv2
import random
from model import UNet
from utils import visualize_prediction
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.functional import dice

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = UNet(backbone="resnet34").to(device)
model.load_state_dict(torch.load("unet_model2.pth", map_location=device, weights_only=True))
model.eval()

# Transformación para normalizar imágenes
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte a [C, H, W] y normaliza a [0,1]
])

# Ruta de test
test_path = "test"
test_images = sorted([f for f in os.listdir(test_path) if f.endswith(".npy")])

# Seleccionar 5 imágenes aleatorias para visualización
random.seed(42)  # Asegurar reproducibilidad
random_images = set(random.sample(test_images, min(5, len(test_images))))

# Inicializar métricas
iou_metric = BinaryJaccardIndex().to(device)
precision_metric = BinaryPrecision().to(device)
recall_metric = BinaryRecall().to(device)
f1_metric = BinaryF1Score().to(device)

# Almacenar resultados
iou_scores, dice_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []


def crop_image_with_transparency(image, mask):
    """Recorta la imagen usando la máscara predicha y hace el fondo transparente."""
    mask = (mask > 127).astype(np.uint8)  # Binarizar la máscara (0 o 1)

    # Convertir la imagen a formato RGBA (añadir canal Alpha)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Asignar la máscara al canal Alpha (255 donde hay lesión, 0 en el fondo)
    image_rgba[:, :, 3] = mask * 255

    return image_rgba


for test_file in test_images:  # Procesar toda la data
    image = np.load(os.path.join(test_path, test_file))  # [H, W, C]

    # Cargar ground truth si lo tienes
    mask_gt_file = test_file.replace(".npy", "_mask.npy")  # Asegura el sufijo correcto
    mask_gt_path = os.path.join("masks_test", mask_gt_file)

    if os.path.exists(mask_gt_path):
        mask_gt = np.load(mask_gt_path)  # [H, W]
    else:
        print(f"⚠️ No se encontró ground truth para {test_file}, omitiendo métricas.")
        continue

    # Convertir a tensor
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
    mask_gt_tensor = (torch.tensor(mask_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0).long()

    # Predicción
    with torch.no_grad():
        pred_mask = model(image_tensor)  # [1, 1, H, W]
        pred_mask_bin = (pred_mask > 0.5).float()  # Binarizar la predicción

    # Convertir a formato numpy para procesamiento
    pred_mask_np = (pred_mask_bin.cpu().numpy()[0, 0] * 255).astype(np.uint8)

    # Recorte de la imagen usando la máscara predicha
    cropped_image = crop_image_with_transparency(image, pred_mask_np)

    # Calcular métricas
    iou = iou_metric(pred_mask_bin, mask_gt_tensor).item()
    dice_score = dice(pred_mask_bin, mask_gt_tensor).item()
    precision = precision_metric(pred_mask_bin, mask_gt_tensor).item()
    recall = recall_metric(pred_mask_bin, mask_gt_tensor).item()
    f1 = f1_metric(pred_mask_bin, mask_gt_tensor).item()

    # Almacenar resultados
    iou_scores.append(iou)
    dice_scores.append(dice_score)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Solo mostrar imágenes para las 5 seleccionadas aleatoriamente
    if test_file in random_images:
        print(f"\n📌 Resultados para {test_file}:")
        print(f"   - IoU: {iou:.4f}")
        print(f"   - Dice Score: {dice_score:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")

        # Guardar la imagen resultante con la imagen recortada
        visualize_prediction(image, mask_gt, pred_mask_np, cropped_image, filename=f"result_{test_file}.png")

# Promedio final de métricas
if iou_scores:
    print("\n📊 **Promedio de métricas en todo el dataset de test:**")
    print(f"   - IoU Promedio: {np.mean(iou_scores):.4f}")
    print(f"   - Dice Score Promedio: {np.mean(dice_scores):.4f}")
    print(f"   - Precision Promedio: {np.mean(precision_scores):.4f}")
    print(f"   - Recall Promedio: {np.mean(recall_scores):.4f}")
    print(f"   - F1-Score Promedio: {np.mean(f1_scores):.4f}")
