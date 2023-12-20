import pickle
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb')) #otvaramo dictionary sa koordinatama landmarkova
#iz dictonary-a izvlacimo koordinate i labele i pretvaramo ih u nizove
max_length = 42 #ouput mediapipe hands objekta su 21 landmark * 2 koordinate. Ako modelu proslijedimo vise ili manje od ovoga, program puca
data = np.asarray([seq[:max_length] + [0.0] * max(0, max_length - len(seq)) for seq in data_dict['data']]) #svaki clan dataseta mora imati niz od 42 elementa. Ako ima vise, krati se na 42 elementa, ako ima manje paduje se do 42
#data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
#razdvajanje dataseta na training (80%) i test (20%). mijesamo podatke i dijelimo ih proporcionalno po labelima
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
model = RandomForestClassifier() #inicijalizujemo objekat klase RandomForestClassifier
model.fit(x_train, y_train) #prosledjujemo koordinate za training
y_predict = model.predict(x_test) 
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f) #cuvamo model
f.close()
