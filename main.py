import cv2 as cv
import mediapipe as mp
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras





import pandas as pd
import os
from datetime import date, datetime

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

#загрузка нашей модели
reconstructed_model = keras.models.load_model("my_model.keras")

cap_device = 0
cap_width = 1920
cap_height = 1080
a = True
use_brect = True

file_path = "data/videoplayback.mp4"

cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
#cap = cv.VideoCapture(file_path)
window_name = 'Мониторинг фокусировки внимания'

cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(window_name, cv.WINDOW_FULLSCREEN, cv.WINDOW_FULLSCREEN)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

log_file_path = os.getcwd() + '\\logs\\' + str(date.today()) + '.csv'

if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as myfile:
        myfile.write('attention detection service log\n')
c = 0

while True:
    # Process Key (ESC: end)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    if results.multi_face_landmarks is not None:
        face_id = 0 
        face_scores = {}
        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)
            
            
            lm_list = []
            #print('face_landmarks:', (face_landmarks))
            #print(len(face_landmarks.landmark))
            for lm in face_landmarks.landmark:
                lm_list.append(lm.x)
                lm_list.append(lm.y)
                lm_list.append(lm.z)

            df = pd.DataFrame(columns=np.arange(478 * 3))
            df.loc[len(df)] = lm_list

            face_array = np.asarray(df)

            model_predict = reconstructed_model.predict(face_array, verbose = False)
            pred = model_predict[0][0].round(3)

            color = (0,0,0)
            if pred > 0.8:
                color = (56,46,223)
            else:
                color = (85,186,85)

            cv.putText(debug_image, str(face_id), ((brect[0] + brect[2]) // 2, brect[1] - 5), cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1, cv.LINE_AA)
            cv.putText(debug_image, str(face_id) + ' : ' + str(pred), (0, 20 + face_id * 15), cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1, cv.LINE_AA)
            # Drawing part
            debug_image = cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), color, 2)

            face_id += 1
            face_scores[face_id] = pred > 0.5

            c+=1
            #print(c)

    if c >= 120:
        with open(log_file_path, "a") as myfile:
            now = datetime.now()
            myfile.write(str(now) + ';' + str(face_scores) + '\n')
        cv.imwrite(log_file_path.split('.')[0] + '-' + str(now.hour) + '-' + str(now.minute)+ '-'  + str(now.second) + '.jpg', debug_image) 
        c = 0

    #cv.imshow('window', debug_image)

    cv.namedWindow(window_name)
    cv.setWindowProperty(window_name, cv.WINDOW_FULLSCREEN, cv.WINDOW_KEEPRATIO)

    cv.imshow(window_name, debug_image)

cap.release()
cv.destroyAllWindows()