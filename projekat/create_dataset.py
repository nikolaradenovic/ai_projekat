import os
import pickle

import mediapipe as mp
import cv2 #open cv
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)# hands je objekat madiapipe Hands klase
#hand tracking se primjenjuje za staticku sliku, prag detekcije ruke je 0.3
DATA_DIR = './data' #folder za smijestanje foldera koje sadrze slike za dataset

data = [] 
labels = [] #klase
for dir_ in os.listdir(DATA_DIR): #iteracija svih foldera u ./data
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #iteracija svih slika u dir_
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) #trenutnu sliku smijestamo u img promjenljivu
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #img je u BGR, pretvaramo je u RGB

        results = hands.process(img_rgb) #detektovanje landmarkova iz slike
        if results.multi_hand_landmarks: #da li je vise landmarkova detektovano. ako jeste, detektovana je barem jedna saka
            for hand_landmarks in results.multi_hand_landmarks: #iteracija landmarkova
                for i in range(len(hand_landmarks.landmark)): #prolazak kroz niz u kojem su koordinate landmarka
                    x = hand_landmarks.landmark[i].x #stavljanje koordinata landmarka u promjenljivu x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x) #appendovanje koordinate listi koordinata
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux) #u data se dodaje data_aux
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f) #data se pridruzuje label i od toga se pravi dictionary 
f.close()
