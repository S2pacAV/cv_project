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
from models.unet_segmentation import UNetSegmentator

# Настройка страницы
st.set_page_config(
    page_title="Семантическая сегментация",
    page_icon="🗺️",
    layout="wide"
)

# Заголовок страницы
st.title("Семантическая сегментация аэрокосмических снимков")

# Инициализация сегментатора
@st.cache_resource
def load_model():
    # Загружаем модель U-Net
    model_path = "/home/s2pac/ElbrusBootcamp/Nikitaproject/prodject/best_model.pth"
    return UNetSegmentator(model_path)

try:
    segmentator = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    model_loaded = False

if model_loaded:
    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        
        # Настройка прозрачности маски
        alpha = st.slider("Прозрачность маски", 0.0, 1.0, 0.5, 0.1, 
                         help="Чем больше значение, тем более видимой будет маска")
        
        # Информация о модели
        st.header("Информация о модели")
        info = segmentator.get_training_info()
        
        st.markdown(f"""
        - **Число эпох обучения:** {info['epochs']}
        - **Размер изображений:** {info['image_size']}x{info['image_size']}
        - **Архитектура:** U-Net
        """)
        
        st.subheader("Объем выборок:")
        st.markdown(f"""
        - Обучающая: {info['dataset_size']['train']} изображений
        - Валидационная: {info['dataset_size']['val']} изображений
        """)
        
        st.subheader("Метрики:")
        st.markdown(f"""
        - Dice Score: {info['best_metrics']['Dice']:.3f}
        - IoU: {info['best_metrics']['IoU']:.3f}
        - Accuracy: {info['best_metrics']['Accuracy']:.3f}
        - Loss: {info['best_metrics']['Loss']:.3f}
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
                    
                    # Семантическая сегментация
                    original, overlay, mask = segmentator.segment_from_image(image, alpha)
                    
                    # Отображение результатов
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Исходное изображение")
                        st.image(original, use_container_width=True)
                    
                    with col2:
                        st.subheader("Предсказанная маска")
                        # Преобразуем маску в формат для отображения (серая маска)
                        mask_display = (mask * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_display, mode='L')
                        st.image(mask_image, use_container_width=True)
                    
                    with col3:
                        st.subheader("Результат сегментации")
                        st.image(overlay, use_container_width=True)
                    
                    # Статистика по маске
                    total_pixels = mask.shape[0] * mask.shape[1]
                    segmented_pixels = np.sum(mask > 0.5)
                    percentage = (segmented_pixels / total_pixels) * 100
                    
                    st.info(f"""
                    **Статистика сегментации:**
                    - Всего пикселей: {total_pixels:,}
                    - Сегментированных пикселей: {segmented_pixels:,}
                    - Процент сегментированной области: {percentage:.2f}%
                    """)
                    
                    st.markdown("---")

    # Использование URL
    with tab2:
        st.header("Сегментация по URL изображения")
        
        # Примеры изображений
        example_images = {
            "Космический снимок 1": "https://russian.news.cn/photo/2016-12/06/135883913_14809906178831n.jpg",
            "Аэрофотоснимок": "https://images.unsplash.com/photo-1602526430717-e0c8e838e885?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
            "Спутниковый снимок": "https://images.unsplash.com/photo-1446776653964-20c1d3a81b06?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"
        }
        
        # selected_example = st.selectbox(
        #     "Выберите пример изображения или введите свой URL", 
        #     ["Выберите пример..."] + list(example_images.keys())
        # )
        
        # if selected_example != "Выберите пример...":
        #     url = example_images[selected_example]
        # else:
        url = st.text_input("Введите URL изображения", "https://img.freepik.com/free-photo/aerial-shot-field-with-colorful-trees-forest_181624-30988.jpg?semt=ais_hybrid&w=740")
        
        if url and st.button("Обработать изображение"):
            try:
                with st.spinner("Загрузка и обработка изображения..."):
                    # Семантическая сегментация по URL
                    original, overlay, mask = segmentator.segment_from_url(url, alpha)
                    
                    # Отображение результатов
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader("Исходное изображение")
                        st.image(original, use_container_width=True)
                    
                    with col2:
                        st.subheader("Предсказанная маска")
                        # Преобразуем маску в формат для отображения (серая маска)
                        mask_display = (mask * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_display, mode='L')
                        st.image(mask_image, use_container_width=True)
                    
                    with col3:
                        st.subheader("Результат сегментации")
                        st.image(overlay, use_container_width=True)
                    
                    # Статистика по маске
                    total_pixels = mask.shape[0] * mask.shape[1]
                    segmented_pixels = np.sum(mask > 0.5)
                    percentage = (segmented_pixels / total_pixels) * 100
                    
                    st.info(f"""
                    **Статистика сегментации:**
                    - Всего пикселей: {total_pixels:,}
                    - Сегментированных пикселей: {segmented_pixels:,}
                    - Процент сегментированной области: {percentage:.2f}%
                    """)
                    
            except Exception as e:
                st.error(f"Ошибка при обработке изображения: {e}")

    # Метрики и графики
    with tab3:
        st.header("Метрики модели")
        
        # График метрик обучения
        metrics_fig = segmentator.plot_metrics_demo()
        st.pyplot(metrics_fig)
        
        st.subheader("О семантической сегментации")
        st.markdown("""
        **Семантическая сегментация** - это задача компьютерного зрения, которая заключается в 
        классификации каждого пикселя изображения в определенную категорию.
        
        В данном приложении используется архитектура **U-Net**:
        
        1. **Энкодер (Encoder)** - сжимает изображение, извлекая важные признаки
        2. **Декодер (Decoder)** - восстанавливает пространственное разрешение
        3. **Skip-connections** - сохраняют детали на разных уровнях разрешения
        
        **Применения:**
        - Анализ спутниковых снимков
        - Медицинская диагностика
        - Автономное вождение
        - Обработка аэрофотоснимков
        
        **Метрики качества:**
        - **Dice Score** - мера схожести между предсказанной и истинной маской
        - **IoU (Intersection over Union)** - отношение пересечения к объединению
        - **Accuracy** - доля правильно классифицированных пикселей
        """)
        
else:
    st.warning("Модель не загружена. Проверьте путь к файлу модели.") 