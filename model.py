import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, backbone="resnet34"):
        super(UNet, self).__init__()

        if backbone == "resnet34":
            base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool

            self.encoder1 = base_model.layer1
            self.encoder2 = base_model.layer2
            self.encoder3 = base_model.layer3
            self.encoder4 = base_model.layer4

            filters = [64, 128, 256, 512]

        elif backbone == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.encoder = base_model.features
            filters = [16, 24, 40, 112]
        else:
            raise ValueError("Backbone no soportado. Usa 'resnet34' o 'efficientnet_b0'.")

        self.up1 = self._upsample(filters[3], filters[2])
        self.up2 = self._upsample(filters[2], filters[1])
        self.up3 = self._upsample(filters[1], filters[0])
        self.up4 = self._upsample(filters[0], 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x1)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        d1 = self.up1(x5)
        d1 = nn.functional.interpolate(d1, size=x4.shape[2:], mode="bilinear", align_corners=False)
        d1 = d1 + x4  # 🔥 Se aseguró la misma dimensión antes de sumar

        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d2 = d2 + x3  # 🔥 Corrección de tamaño

        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d3 = d3 + x2  # 🔥 Corrección de tamaño

        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(d4, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d4 = d4 + x1  # 🔥 Corrección de tamaño

        x = self.final_conv(d4)
        x = nn.functional.interpolate(x, size=(450, 600), mode="bilinear", align_corners=False)

        return x
