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
        image = cv.flip(image, 1)  # Отзеркаливаем изображение для удобства
    debug_image = copy.deepcopy(image)  # Создаем копию изображения для отладки

    # Обработка изображения для обнаружения лиц
    results = face_mesh.process(image)  # Применяем модель для нахождения точек лица
    
    if results.multi_face_landmarks is not None:  # Проверяем, найдены ли лица
        face_id = 0  # Идентификатор лица
        face_scores = {}  # Словарь для хранения оценок лиц
        for face_landmarks in results.multi_face_landmarks:  # Проходим по всем найденным лицам
            # Вычисляем ограничивающий прямоугольник вокруг лица
            brect = calc_bounding_rect(debug_image, face_landmarks)
            
            lm_list = []  # Список для хранения координат точек лица
            for lm in face_landmarks.landmark:  # Проходим по всем точкам лица
                lm_list.append(lm.x)  # Добавляем x-координату
                lm_list.append(lm.y)  # Добавляем y-координату
                lm_list.append(lm.z)  # Добавляем z-координату

            df = pd.DataFrame(columns=np.arange(478 * 3))  # Создаем DataFrame для координат
            df.loc[len(df)] = lm_list  # Записываем координаты в DataFrame

            face_array = np.asarray(df)  # Преобразуем DataFrame в массив NumPy

            # Прогнозируем с помощью загруженной модели
            model_predict = reconstructed_model.predict(face_array, verbose=False)
            pred = model_predict[0][0].round(3)  # Получаем предсказание и округляем

            # Определяем цвет текста в зависимости от предсказания
            color = (0, 0, 0)  # По умолчанию черный цвет
            if pred > 0.8:  # Если вероятность высокая
                color = (56, 46, 223)  # Цвет для высокого внимания
            else:  # Если вероятность низкая
                color = (85, 186, 85)  # Цвет для низкого внимания

            # Добавляем текст на изображение
            cv.putText(debug_image, str(face_id), ((brect[0] + brect[2]) // 2, brect[1] - 5), cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1, cv.LINE_AA)
            cv.putText(debug_image, str(face_id) + ' : ' + str(pred), (0, 20 + face_id * 15), cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1, cv.LINE_AA)
            # Рисуем ограничивающий прямоугольник
            debug_image = cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), color, 2)

            face_id += 1  # Увеличиваем идентификатор лица
            face_scores[face_id] = pred > 0.5  # Сохраняем результат предсказания

            c += 1  # Увеличиваем счетчик

    # Каждые 120 кадров записываем результаты в лог
    if c >= 120:
        with open(log_file_path, "a") as myfile:
            now = datetime.now()  # Получаем текущее время
            myfile.write(str(now) + ';' + str(face_scores) + '\n')  # Записываем время и оценки в лог
        # Сохраняем изображение с отладочной информацией
        cv.imwrite(log_file_path.split('.')[0] + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.jpg', debug_image) 
        c = 0  # Сбрасываем счетчик

    # Настройка окна для отображения изображения
    cv.namedWindow(window_name)
    cv.setWindowProperty(window_name, cv.WINDOW_FULLSCREEN, cv.WINDOW_KEEPRATIO)

    cv.imshow(window_name, debug_image)  # Показываем изображение с отладочной информацией

cap.release()  # Освобождаем видеопоток
cv.destroyAllWindows()  # Закрываем все окна OpenCV

