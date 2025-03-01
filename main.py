import cv2 as cv  # Импортируем библиотеку OpenCV для обработки изображений
import mediapipe as mp  # Импортируем библиотеку MediaPipe для распознавания лиц
import copy  # Импортируем модуль copy для создания копий объектов
import numpy as np  # Импортируем NumPy для работы с массивами
import tensorflow as tf  # Импортируем TensorFlow для работы с моделями машинного обучения
from tensorflow import keras  # Импортируем Keras из TensorFlow для загрузки модели

import pandas as pd  # Импортируем Pandas для работы с данными в формате DataFrame
import os  # Импортируем os для работы с файловой системой
from datetime import date, datetime  # Импортируем модули для работы с датой и временем

# Функция для вычисления ограничивающего прямоугольника вокруг лица
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]  # Получаем размеры изображения

    landmark_array = np.empty((0, 2), int)  # Создаем пустой массив для координат точек

    # Проходим по всем точкам (landmarks) и преобразуем их координаты
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)  # Преобразуем x-координату
        landmark_y = min(int(landmark.y * image_height), image_height - 1)  # Преобразуем y-координату

        landmark_point = [np.array((landmark_x, landmark_y))]  # Создаем массив точки

        landmark_array = np.append(landmark_array, landmark_point, axis=0)  # Добавляем точку в массив

    # Вычисляем ограничивающий прямоугольник
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]  # Возвращаем координаты ограничивающего прямоугольника

# Загрузка нашей модели
reconstructed_model = keras.models.load_model("my_model.keras")

# Настройки для видеозахвата
cap_device = 0  # Устройство захвата (0 - веб-камера)
cap_width = 1920  # Ширина захватываемого изображения
cap_height = 1080  # Высота захватываемого изображения
a = True  # Переменная, возможно, для других условий
use_brect = True  # Использовать ли ограничивающий прямоугольник

file_path = "data/videoplayback.mp4"  # Путь к видеофайлу (не используется в текущем коде)

cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)  # Инициализация видеозахвата
# cap = cv.VideoCapture(file_path)  # Альтернативный способ захвата из видеофайла
window_name = 'Мониторинг фокусировки внимания'  # Название окна

# Настройка окна для отображения
cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(window_name, cv.WINDOW_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# Инициализация модели распознавания лиц
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,  # Максимальное количество распознаваемых лиц
        refine_landmarks=True,  # Улучшение распознавания точек
        min_detection_confidence=0.5,  # Минимальная уверенность для распознавания
        min_tracking_confidence=0.5)  # Минимальная уверенность для отслеживания

# Путь для сохранения логов
log_file_path = os.getcwd() + '\\logs\\' + str(date.today()) + '.csv'

# Если лог-файл не существует, создаем его
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as myfile:
        myfile.write('attention detection service log\n')  # Записываем заголовок

c = 0  # Счетчик для записи логов

while True:  # Основной цикл
    # Обработка нажатия клавиш (ESC: выход)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    ret, image = cap.read()  # Чтение изображения из видеопотока
    if not ret:  # Если не удалось прочитать изображение, выходим
        break
    image =
