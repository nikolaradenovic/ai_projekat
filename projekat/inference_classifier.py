import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb')) #otvaramo model.p
model = model_dict['model'] #u model stavljamo value key-a 'model', tj nas trenirani model
cap = cv2.VideoCapture(0) #pristupamo web kameri
mp_hands = mp.solutions.hands #inicijalizujemo objekat klase hands za pracenje sake
mp_drawing = mp.solutions.drawing_utils #klasa drawing_utils je za crtanje landmarkova i anotaciju
mp_drawing_styles = mp.solutions.drawing_styles #stilovi za crtanje
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #hands je objekat madiapipe Hands klase
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
               9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
               17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6',
               33: '7', 34: '8', 35: '9'} #oznacavanje klasa

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #pretvaranje frejma iz bgr u rgb
    results = hands.process(frame_rgb) #detektovanje landmarkova iz slike
    try: #ako se na slici pojavi vise ili manje od 48 landmarka, program puca. Zbog toga postoji ovaj try/except blok: da bi program samo nastavio
        if results.multi_hand_landmarks: #ako je otkriveno vise landmarka, znaci da je detektovana minimum jedna ruka
            for hand_landmarks in results.multi_hand_landmarks: #iteracija landmarka
                mp_drawing.draw_landmarks( #crtanja landmarka
                    frame,  #na frame se crta
                    hand_landmarks,  #landmarkovi
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(), #stilovi
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks: #iteracija landmarka
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_)) #dodavanje normalizovanih vrijednosti u data_aux
                    data_aux.append(y - min(y_))
                    
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            #print(data_aux)
            prediction = model.predict([np.asarray(data_aux)]) 
            predicted_character = labels_dict[int(prediction[0])] #prediction je lista od jednog elementa, pa taj element smijestamo u promjenljivu prediction  
            prediction_probability = model.predict_proba([np.asarray(data_aux)]) #vector vjerovatnoce svih klasa
            threshold = np.max(prediction_probability)*100 #prag detekcije
            most_likely_character = str((np.max(prediction_probability))*100) + "%" #sigurnost predikcije

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4) #crtanje pravougaonika na slici  
            if threshold > 30: #ako je model barem 30% siguran u svoju predikciju:
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA) #prikazivanje predikcije na slici
                cv2.putText(frame, most_likely_character, (x1+50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA) #prikazivanje sigurnosti predikcije na slici
           
    except:
        continue
    
    cv2.imshow('frame', frame) #prikazujemo frame
    cv2.waitKey(1) #refreshujemo ga svake milisekunde


cap.release() #oslobadjanje memorije
cv2.destroyAllWindows() #zatvaranje svih prozora
