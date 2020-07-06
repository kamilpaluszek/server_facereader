from flask import Flask

from importlib import import_module
import os
import jsonpickle
import json
from flask import Flask, Response , request , flash , url_for,jsonify
import logging
from logging.config import dictConfig
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import json
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import io


#inicjalizacja biblioteki Flask
server = Flask(__name__)

#szerokosc zdjecia
WIDTH = 48
#wysokosc zdjecia
HEIGHT = 48

#x = None
#y = None


#strona, na którą jest przesyłane zapytanie POST w celu klasyfikacji zdjęcia przesłanego przez mobilne urządzenie
@server.route('/classifier/run', methods=['POST'])
#funkcja odpowiedzialna za klasyfikacje przeslanego zdjecia
def classify():
    server.logger.debug('Running classifier')
    #odczyt zdjęcia wysłanego przez mobilne urządzenie
    filestr = request.files['image'].read()
    #zaladowanie modelu
    model = load_model()
    #przetworzenie oraz klasyfikacja zdjęcia mobilnego urządzenie (tablica) [0-parametry kwadratu twarzy, 1-predykcja zdjecia]
    classifyResult = load_image(model, filestr)
    #zapisanie twarzy ze zdjęcia do zmiennej
    imgFaces = classifyResult[0]
    # zapisanie predykcji emocji ze zdjęcia do zmiennej
    prediction = classifyResult[1]
    #sprawdzenie, czy parametry są zapisane w odpowiednim formacie, oraz zapisanie do zmiennej ilości twarzy na zdjęciu
    if(isinstance(imgFaces, np.ndarray)):
        countFaces = int(imgFaces.size/4)
    try:
        arrayJSON = "["
        i = 0

        #iteracja po każdym parametrze twarzy w celu zapisania ich do formatu JSON
        for x,y,w,h in imgFaces:
            #zapis parametrow oraz predykcji do zmiennej wykorzystując format JSON
            jsondump = json.dumps( {"prediction": prediction[i],
            "xrect": np.int64(x), "yrect": np.int64(y),
                            "wrect": np.int64(w), "hrect": np.int64(h)}, default=convert)
            i+=1

            #odpowiednie sformatowanie zakonczenia formatu JSON
            if(countFaces<2):
                arrayJSON += jsondump
            else:
                arrayJSON += jsondump+ ","
        if(arrayJSON[-1] == ","):
            arrayJSON = arrayJSON[:-1] + "]"
        else:
            arrayJSON += "]"
        #jesli dlugosc zmiennej bedzie wynosila 2, to znaczy ze tablica jest pusta (bedzie posiadac tylko "[]"),
        # czyli klasyfikacja się nie powiodła
        if len(arrayJSON) == 2:
            #wyslanie formatu JSON o niezidentyfikowaniu zdjecia
            return json.dumps([{"prediction": "Unable"}])
        print(arrayJSON)
        return arrayJSON

    #obsluga bledu, jesli server nie znalazl zdjecia o podanym parametrze POST
    except FileNotFoundError as e:
        return json.dumps([{"prediction": "File not Found!"}])

    #except Exception as e:
   #     return json.dumps([{"prediction": "nue wuem"}])


#funkcja sprawdzająca, czy parametry są np.int64
def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

#funkcja odpowiedzialna za zaladowanie zdjęcia do modelu
def load_image(model, filename):

    # wyswietlenie zdjecia po stronie servera
   # image_data = filename
   # image = Image.open(io.BytesIO(image_data))
   # image.show()

    #zamiana z bajtowej reprezentacji zdjęcia na tablice np
    npimg = np.fromstring(filename, dtype=np.uint8)

    # zamiana tablicy np na zdjecie o pelnej rozdzielczosci
    full_size_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    #zamiana pelnego zdjecia na zdjecie o szarej skali (nasz model sie na takich zdjeciach uczyl)
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    #odczyt parametrow twarzy za pomoca haarcascade_frontalface
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)
    prediction = np.array([])
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            #zamiana zdjecia na zdjecie 48x48 - takie na jakich nasz model sie uczyl
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predykcja emocji
            yhat = run_model(model, cropped_img)
            predictionlabels = yhat
            #w przypadku wielu twarz - nadpisywanie do zmiennej tablicowej kazdej emocji
            prediction = np.append(prediction, predictionlabels)
            print(prediction)
    return faces, prediction

#wszystkie możliwe emocje
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#funkcja odpowiedzialna za zaladowanie modelu z dysku, na podstawie którego nastąpi predykcja zdjęcia
def load_model():
    try:
        #otwarcie naszego modelu zapisanego w formacie JSON w trybie read
        json_file = open('fer.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #załadowanie wag do załadowanego modelu
        loaded_model.load_weights("fer.h5")
        print("Loaded model from disk")
        return loaded_model
    #obsluga bledu, jesli nie znaleziono modelu o okreslonej nazwie w naszym folderze
    except FileNotFoundError as e:
        return os.abort('Unable to find the file: %s.' % str(e), 503)

#funkcja odpowiedzialna za predykcje na podstawie zaladowanego modelu
def run_model(loaded_model, img):
    #predykcja w formie wartosci dziesietnych kazdej z emocji
    predictions_percentage = loaded_model.predict(img)
    #zapis do zmiennej emocji, ktora ma najwieksza wartosc - najwieksze prawdopodobienstwo, ze to jest poprawny wynik
    prediction = labels[np.argmax(predictions_percentage)]
    return prediction

if __name__ == '__main__':
    server.run(host='0.0.0.0', threaded=False)