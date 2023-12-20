import cv2
import os

DATA_DIR = './data' #folder za smijestanje foldera koje sadrze slike za dataset
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36 #broj klasa, folderi ce se zvati od 0-35
dataset_size = 250 #broj slika u svakom folderu

cap = cv2.VideoCapture(0)
for j in range(number_of_classes): #pravljenje foldera za klase
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    done = False
    while True: #infinite loop koji prikazuje webcam feed i ceka prekid pritiskom dugmeta q
        ret, frame = cap.read()
        cv2.putText(frame, 'Press Q to collect', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0 #brojac koji prati koliko je slika prikupljeno
    while counter < dataset_size: #nakon prekida prosle petlje, pocinje prikupljanje slika  sa feeda. Prikuplja se onoliko slika koliko je definisano sa dataset_size
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1 #uvecavanje brojaca za 1

cap.release()
cv2.destroyAllWindows()
