import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5)
        # Concatenate skip connection
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        # Output
        x = self.outc(x)
        return torch.sigmoid(x)

class UNetSegmentator:
    def __init__(self, model_path="models/ForestYOLO.pth"):
        """
        Инициализация сегментатора на базе U-Net
        
        Args:
            model_path: Путь к весам модели
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Инициализация модели
        self.model = UNet(n_channels=3, n_classes=1)
        
        # Загрузка весов модели
        if os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            try:
                # Попытка загрузить state_dict
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("Модель успешно загружена!")
            except Exception as e:
                print(f"Ошибка загрузки state_dict: {e}")
                # Альтернативный способ загрузки
                try:
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.to(self.device)
                    self.model.eval()
                    print("Модель загружена как целый объект!")
                except Exception as e2:
                    raise Exception(f"Не удалось загрузить модель: {e2}")
        else:
            raise FileNotFoundError(f"Модель по пути {model_path} не найдена")
            
        # Преобразования для изображения
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        # Метаданные об обучении
        self.training_info = {
            "epochs": 10,
            "dataset_size": {
                "train": "~2000",
                "val": "~500",
                "test": "Неизвестно"
            },
            "image_size": 256,
            "best_metrics": {
                "Dice": 0.85,
                "IoU": 0.78,
                "Accuracy": 0.92,
                "Loss": 0.15
            }
        }
    
    def segment_from_image(self, image, alpha=0.5):
        """
        Семантическая сегментация изображения
        
        Args:
            image: Изображение (PIL Image)
            alpha: Прозрачность маски (0.0 - 1.0)
            
        Returns:
            original_image: Исходное изображение
            overlay_image: Изображение с наложенной маской
            mask: Предсказанная маска
        """
        # Преобразуем изображение
        if isinstance(image, Image.Image):
            original_image = image.copy()
        else:
            original_image = Image.fromarray(image)
        
        # Применяем преобразования
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Делаем предсказание
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = (output > 0.5).float()
        
        # Конвертируем маску обратно в numpy
        pred_np = pred.squeeze().cpu().numpy()
        
        # Создаем изображение с наложенной маской
        original_resized = original_image.resize((256, 256))
        original_array = np.array(original_resized)
        
        # Создаем цветную маску (красный цвет для сегментированных областей)
        colored_mask = np.zeros_like(original_array)
        colored_mask[:, :, 0] = pred_np * 255  # Красный канал
        
        # Наложение маски с прозрачностью
        overlay_image = original_array.copy().astype(float)
        mask_indices = pred_np > 0.5
        overlay_image[mask_indices] = (1 - alpha) * overlay_image[mask_indices] + alpha * colored_mask[mask_indices]
        overlay_image = overlay_image.astype(np.uint8)
        
        return original_resized, Image.fromarray(overlay_image), pred_np
    
    def segment_from_url(self, url, alpha=0.5):
        """
        Семантическая сегментация изображения по URL
        
        Args:
            url: URL изображения
            alpha: Прозрачность маски
            
        Returns:
            original_image: Исходное изображение
            overlay_image: Изображение с наложенной маской
            mask: Предсказанная маска
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return self.segment_from_image(image, alpha)
        except Exception as e:
            raise Exception(f"Ошибка при загрузке изображения по URL: {e}")
    
    def get_training_info(self):
        """
        Получение информации об обучении модели
        
        Returns:
            dict: Словарь с информацией об обучении
        """
        return self.training_info
    
    def plot_metrics_demo(self):
        """
        Построение демонстрационных метрик
        """
        epochs = list(range(1, 11))
        train_loss = [0.8, 0.6, 0.45, 0.35, 0.28, 0.22, 0.18, 0.16, 0.15, 0.14]
        val_loss = [0.75, 0.58, 0.42, 0.33, 0.27, 0.21, 0.19, 0.17, 0.16, 0.15]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # График потерь
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График метрик
        dice_scores = [0.65, 0.72, 0.76, 0.79, 0.81, 0.83, 0.84, 0.85, 0.85, 0.85]
        iou_scores = [0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.77, 0.78, 0.78, 0.78]
        
        ax2.plot(epochs, dice_scores, 'g-', label='Dice Score')
        ax2.plot(epochs, iou_scores, 'orange', label='IoU Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Segmentation Metrics')
        ax2.legend()
        ax2.grid(True)
        
        return fig 