import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

class ShipDetector:
    def __init__(self, model_path= "models/ShipYOLO.pt"):
        """
        Инициализация детектора судов на базе YOLO
        
        Args:
            model_path: Путь к весам модели. Если None, будет использована предобученная модель YOLOv8n
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Если путь к модели не указан, используем предобученную модель YOLOv8
        if model_path is None or not os.path.exists(model_path):
            print(f"Используется моя обученная модель")
            self.model = YOLO('models/ShipYOLO.pt')
        else:
            print(f"Загрузка модели из {model_path}")
            self.model = YOLO(model_path)
            
        # Метаданные об обучении
        self.training_info = {
            "epochs": 30,
            "dataset_size": {
                "train": 9697,
                "val": 2165,
                "test": "Неизвестно"
            },
            "image_size": 320,
            "best_metrics": {
                "mAP50": 0.429,
                "mAP50-95": 0.25,
                "precision": 0.501,
                "recall": 0.435
            }
        }
    
    def detect_from_image(self, image, conf_threshold=0.3):
        """
        Детекция судов на изображении
        
        Args:
            image: Изображение (PIL Image, numpy array, путь к файлу)
            conf_threshold: Порог уверенности для детекции
            
        Returns:
            processed_image: Изображение с отмеченными боксами (в формате RGB)
            results: Результат работы модели
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Конвертируем BGR → RGB перед возвратом
        plotted_image = results[0].plot()
        rgb_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        return rgb_image, results[0]
    
    def detect_from_url(self, url, conf_threshold=0.3):
        """
        Детекция судов на изображении по URL
        
        Args:
            url: URL изображения
            conf_threshold: Порог уверенности для детекции
            
        Returns:
            processed_image: Изображение с отмеченными боксами (в формате RGB)
            results: Результат работы модели
        """
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            return self.detect_from_image(image, conf_threshold)
        except Exception as e:
            raise Exception(f"Ошибка при загрузке изображения по URL: {e}")
    
    def get_training_info(self):
        """
        Получение информации об обучении модели
        
        Returns:
            dict: Словарь с информацией об обучении
        """
        return self.training_info
    
    def plot_pr_curve(self):
        """
        Построение PR-кривой (для демонстрации)
        """
        # Демонстрационные данные для PR-кривой
        precision = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.88, 0.92, 0.95, 0.98, 0.99])
        recall = np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('PR Curve for Ship Detection', fontsize=16)
        ax.grid(True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        return fig
    
    def plot_confusion_matrix(self):
        """
        Построение матрицы ошибок (для демонстрации)
        """
        # Демонстрационные данные для матрицы ошибок
        conf_matrix = np.array([[3400, 320], [450, 3050]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        fig.colorbar(im)
        
        classes = ['Нет судна', 'Судно']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Подписи значений в ячейках
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        return fig 
    
    
    
