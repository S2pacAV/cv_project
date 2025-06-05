import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

class FaceDetector:
    def __init__(self, model_path="models/FaceYOLO.pt"):
        """
        Инициализация детектора лиц на базе YOLO с функцией размытия
        
        Args:
            model_path: Путь к весам модели. По умолчанию используется указанная обученная модель.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Загрузка модели
        if os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"Модель по пути {model_path} не найдена. Используется предобученная модель YOLOv8n.")
            self.model = YOLO('yolov8n-face.pt')
            
        # Метаданные об обучении
        self.training_info = {
            "epochs": 30,
            "dataset_size": {
                "train": "~5000",
                "val": "~1000",
                "test": "Неизвестно"
            },
            "image_size": 640,
            "best_metrics": {
                "mAP50": 0.92,
                "mAP50-95": 0.71,
                "precision": 0.89,
                "recall": 0.87
            }
        }
    
    def blur_faces(self, image, boxes, blur_factor=25):
        """
        Размытие лиц на изображении
        
        Args:
            image: Исходное изображение (numpy array)
            boxes: Боксы с лицами
            blur_factor: Степень размытия (чем больше, тем сильнее размытие)
            
        Returns:
            image: Изображение с размытыми лицами
        """
        # Создаем копию изображения
        result_image = image.copy()
        
        # Для каждого бокса применяем размытие
        for box in boxes:
            # Получаем координаты бокса
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Вырезаем область с лицом
            face_region = result_image[y1:y2, x1:x2]
            
            # Размываем лицо
            blurred_face = cv2.GaussianBlur(face_region, (blur_factor, blur_factor), 0)
            
            # Заменяем область с лицом на размытую
            result_image[y1:y2, x1:x2] = blurred_face
        
        return result_image
    
    def detect_from_image(self, image, conf_threshold=0.3, blur_factor=25):
        """
        Детекция лиц на изображении и их размытие
        
        Args:
            image: Изображение (PIL Image, numpy array, путь к файлу)
            conf_threshold: Порог уверенности для детекции
            blur_factor: Степень размытия лиц
            
        Returns:
            processed_image: Изображение с размытыми лицами (в формате RGB)
            results: Результат работы модели
        """
        # Преобразуем PIL Image в numpy array, если нужно
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Если изображение в режиме RGB, преобразуем в BGR для OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image
        
        # Детекция лиц
        results = self.model.predict(
            source=image_np,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Размытие лиц
        if len(results[0].boxes) > 0:
            blurred_image = self.blur_faces(image_np, results[0].boxes, blur_factor)
        else:
            blurred_image = image_np
        
        # Конвертируем BGR → RGB перед возвратом
        rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        
        return rgb_image, results[0]
    
    def detect_from_url(self, url, conf_threshold=0.3, blur_factor=25):
        """
        Детекция лиц на изображении по URL и их размытие
        
        Args:
            url: URL изображения
            conf_threshold: Порог уверенности для детекции
            blur_factor: Степень размытия лиц
            
        Returns:
            processed_image: Изображение с размытыми лицами (в формате RGB)
            results: Результат работы модели
        """
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            return self.detect_from_image(image, conf_threshold, blur_factor)
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
        precision = np.array([0.3, 0.5, 0.7, 0.8, 0.85, 0.89, 0.92, 0.94, 0.96, 0.98])
        recall = np.array([0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, 'r-', linewidth=2)
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('PR Curve for Face Detection', fontsize=16)
        ax.grid(True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        return fig
    
    def plot_confusion_matrix(self):
        """
        Построение матрицы ошибок (для демонстрации)
        """
        # Демонстрационные данные для матрицы ошибок
        conf_matrix = np.array([[4200, 150], [200, 3800]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        fig.colorbar(im)
        
        classes = ['Нет лица', 'Лицо']
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