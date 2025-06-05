import streamlit as st
import os
import sys
import numpy as np
import cv2
from PIL import Image
import io
import time
import tempfile
import matplotlib.pyplot as plt

# Добавляем путь к корню проекта для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.face_detector import FaceDetector

# Настройка страницы
st.set_page_config(
    page_title="Детекция лиц с размытием",
    page_icon="👤",
    layout="wide"
)

# Заголовок страницы
st.title("Детекция лиц с размытием")

# Инициализация детектора
@st.cache_resource
def load_model():
    # Загружаем модель детекции лиц
    model_path = "/home/s2pac/ElbrusBootcamp/ZhenyaProject/runs/detect/train/weights2/best.pt"
    return FaceDetector(model_path)

detector = load_model()

# Боковая панель с настройками
with st.sidebar:
    st.header("Настройки")
    
    # Настройка порога уверенности
    confidence = st.slider("Порог уверенности", 0.0, 1.0, 0.3, 0.05)
    
    # Настройка степени размытия
    blur_amount = st.slider("Степень размытия", 5, 45, 25, 2, 
                           help="Чем больше значение, тем сильнее размытие лиц")
    
    # Информация о модели
    st.header("Информация о модели")
    info = detector.get_training_info()
    
    st.markdown(f"""
    - **Число эпох обучения:** {info['epochs']}
    - **Размер изображений:** {info['image_size']}x{info['image_size']}
    """)
    
    st.subheader("Объем выборок:")
    st.markdown(f"""
    - Обучающая: {info['dataset_size']['train']} изображений
    - Валидационная: {info['dataset_size']['val']} изображений
    """)
    
    st.subheader("Метрики:")
    st.markdown(f"""
    - mAP50: {info['best_metrics']['mAP50']:.3f}
    - mAP50-95: {info['best_metrics']['mAP50-95']:.3f}
    - Точность: {info['best_metrics']['precision']:.3f}
    - Полнота: {info['best_metrics']['recall']:.3f}
    """)

# Основная часть страницы
tab1, tab2, tab3 = st.tabs(["Загрузка файлов", "URL изображения", "Метрики"])

# Загрузка изображений
with tab1:
    st.header("Загрузка изображений")
    
    uploaded_files = st.file_uploader("Выберите одно или несколько изображений", 
                                      type=["jpg", "jpeg", "png"], 
                                      accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Отображаем прогресс
            with st.spinner(f"Обработка изображения {uploaded_file.name}..."):
                # Загрузка изображения
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Детекция лиц и размытие
                result_image, result = detector.detect_from_image(
                    image, 
                    conf_threshold=confidence, 
                    blur_factor=blur_amount
                )
                
                # Отображение результатов
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Исходное изображение")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("Результат детекции с размытием")
                    st.image(result_image, use_container_width=True)
                
                # Информация о детекциях
                if len(result.boxes) > 0:
                    st.success(f"Обнаружено лиц: {len(result.boxes)}")
                    
                    # Таблица с детекциями
                    boxes_data = []
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        boxes_data.append({
                            "№": i+1,
                            "Уверенность": f"{conf:.2f}",
                            "Координаты (xmin, ymin, xmax, ymax)": 
                                f"{box.xyxy[0][0]:.1f}, {box.xyxy[0][1]:.1f}, {box.xyxy[0][2]:.1f}, {box.xyxy[0][3]:.1f}"
                        })
                    
                    st.table(boxes_data)
                else:
                    st.warning("Лица не обнаружены")
                
                st.markdown("---")

# Использование URL
with tab2:
    st.header("Детекция по URL изображения")
    
    url = st.text_input("Введите URL изображения", 
                         "https://thumbs.dreamstime.com/z/%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D0%B0-%D0%BB%D1%8E%D0%B4%D0%B5%D0%B9-%D1%80%D0%B0%D0%B7%D0%BD%D0%BE%D0%B3%D0%BE-%D0%B2%D0%BE%D0%B7%D1%80%D0%B0%D1%81%D1%82%D0%B0-%D0%BA%D0%B0%D0%BA-%D1%87%D0%B0%D1%81%D1%82%D0%BD%D1%8B%D0%B5-%D0%BB%D0%B8%D1%86%D0%B0-%D1%80%D0%B0%D0%B7%D0%BD%D1%8B%D1%85-%D0%B2%D0%BE%D0%B7%D1%80%D0%B0%D1%81%D1%82%D0%BE%D0%B2-218504977.jpg")
    
    if url and st.button("Обработать изображение"):
        try:
            with st.spinner("Загрузка и обработка изображения..."):
                # Детекция лиц по URL и размытие
                result_image, result = detector.detect_from_url(
                    url, 
                    conf_threshold=confidence, 
                    blur_factor=blur_amount
                )
                
                # Отображение результатов
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Исходное изображение")
                    st.image(url, use_container_width=True)
                
                with col2:
                    st.subheader("Результат детекции с размытием")
                    st.image(result_image, use_container_width=True)
                
                # Информация о детекциях
                if len(result.boxes) > 0:
                    st.success(f"Обнаружено лиц: {len(result.boxes)}")
                    
                    # Таблица с детекциями
                    boxes_data = []
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        boxes_data.append({
                            "№": i+1,
                            "Уверенность": f"{conf:.2f}",
                            "Координаты (xmin, ymin, xmax, ymax)": 
                                f"{box.xyxy[0][0]:.1f}, {box.xyxy[0][1]:.1f}, {box.xyxy[0][2]:.1f}, {box.xyxy[0][3]:.1f}"
                        })
                    
                    st.table(boxes_data)
                else:
                    st.warning("Лица не обнаружены")
                
        except Exception as e:
            st.error(f"Ошибка при обработке изображения: {e}")

# Метрики и графики
with tab3:
    st.header("Метрики модели")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PR-кривая")
        pr_curve = detector.plot_pr_curve()
        st.pyplot(pr_curve)
        
        st.markdown("""
        **PR-кривая** (Precision-Recall) показывает соотношение между точностью и полнотой модели 
        при различных порогах уверенности. Идеальная модель имеет кривую, которая проходит через 
        правый верхний угол графика.
        """)
    
    with col2:
        st.subheader("Матрица ошибок")
        conf_matrix = detector.plot_confusion_matrix()
        st.pyplot(conf_matrix)
        
        st.markdown("""
        **Матрица ошибок** показывает количество истинно положительных, 
        ложно положительных, истинно отрицательных и ложно отрицательных результатов.
        """)
        
    st.subheader("О размытии лиц")
    st.markdown("""
    Размытие лиц (face blurring) - это техника, которая используется для защиты приватности людей на изображениях.
    В данном приложении:
    
    1. Сначала мы детектируем лица с помощью модели YOLO
    2. Затем применяем фильтр Гауссовского размытия к каждой обнаруженной области лица
    3. Степень размытия можно регулировать с помощью ползунка в боковой панели
    
    Эта техника часто применяется в системах видеонаблюдения, журналистике и социальных медиа для 
    защиты личных данных и соблюдения законов о приватности.
    """)