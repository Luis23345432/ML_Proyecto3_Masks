import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SkinLesionDataset
from model import UNet
from tqdm import tqdm
import os

# 🔥 Configuración del dispositivo y optimización de CUDNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Mejora velocidad en GPU
torch.backends.cudnn.enabled = True  # Asegura uso de CUDNN

# Rutas de dataset
image_path = "pre_processing"
mask_path = "masks_pre_processing"

# Hiperparámetros optimizados
BATCH_SIZE = 16  # Ajusta según VRAM
EPOCHS = 20
LEARNING_RATE = 3e-4  # 🔥 Aumenté LR para convergencia más rápida
ACCUMULATION_STEPS = 2  # 🔥 Simula batch_size=32 en VRAM baja


def train():
    # Cargar dataset
    train_set = SkinLesionDataset(image_path, mask_path)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Modelo
    model = UNet(backbone="resnet34").to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # 🔥 AdamW mejora generalización
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)  # 🔥 Ajuste dinámico de LR
    scaler = torch.amp.GradScaler("cuda")  # 🔥 Corrección de FutureWarning

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        optimizer.zero_grad()

        for i, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):  # 🔥 Mixed Precision
                outputs = model(images)
                loss = criterion(outputs, masks) / ACCUMULATION_STEPS  # 🔥 Divide pérdida acumulada

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)  # 🔥 Escalar pérdida mostrada

        scheduler.step()  # 🔥 Ajuste dinámico de LR
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), "unet_model.pth")
    print("✅ Modelo guardado exitosamente.")

# 🔥 Solución al problema de multiprocessing en Windows
if __name__ == '__main__':
    train()

