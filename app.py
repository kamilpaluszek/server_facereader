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



app = Flask(__name__)

WIDTH = 48
HEIGHT = 48
x = None
y = None




#detecting faces


@app.route('/dre')
def hello():
    return "Hello World!"



@app.route('/test/',methods=['POST'])
def handle_request():
    return "Flask Server & Android are Working Successfully"



@app.route('/classifier/run',methods=['POST'])
def classify():
    app.logger.debug('Running classifier')
    model = load_model()
    # read image file string data
    filestr = request.files['image'].read()

    #load_image() is to process image :
    result = load_image(model, filestr)
    #image = result[0]

   # image = result
    #xrect = result[1]
    #yrect = result[2]
    #hrect=result[4]
   # wrect=result[3]

    facials = result[0]
    prediction = result[1]
    #print(facials)
    #print(type(facials))
    if(isinstance(facials, np.ndarray)):
        countFacials = int(facials.size/4)
    #xrej = json.dumps([{"Faces": countFacials}])


    print('image ready')
    try:

        #prediction = run_model(image)


       # xrecter = result[1]
       # print(type(xrecter))

        #array = json.dumps({"prediction": prediction})
        xrej = "["
        i = 0




        for x,y,w,h in facials:

            array = json.dumps( {"prediction": prediction[i],
            "xrect": np.int64(x), "yrect": np.int64(y),
                            "wrect": np.int64(w), "hrect": np.int64(h)}, default=convert)
            i+=1
            if(countFacials<2):
            # if(facials<2):
                xrej += array
            else:
                xrej += array+ ","
        if(xrej[-1] == ","):
            xrej = xrej[:-1] + "]"
        else:
            xrej += "]"

        if len(xrej) == 2:
            return json.dumps([{"prediction": "nie wiem co sie odkriwa"}])
        print(xrej)
        return xrej



    except FileNotFoundError as e:
        return json.dumps([{"prediction": "File not Found!"}])
    except Exception as e:
        return json.dumps([{"prediction": "nue wuem"}])


def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


def load_image(model, filename):
    # loading image

    # setting image resizing parameters

    # convert string data to numpy array
    #image_data = filename  # byte values of the image
    #image = Image.open(io.BytesIO(image_data))
    #image.show()

    npimg = np.fromstring(filename, dtype=np.uint8)

    # convert numpy array to image
    full_size_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    #print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)
    predictio = np.array([])
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            #print(x, y, w, h)

        #predicting the emotion
            yhat = run_model(model, cropped_img)

            prediction = yhat
            predictio = np.append(predictio, prediction)
            print(predictio)

          ## 1, cv2.LINE_AA)
  #  cv2.imshow('Emotion', full_size_image )
   # cv2.waitKey()

    #multiple faces i teraz jakos zdumpowac kilka wartosci ?
    return faces, predictio#, x, y, w, h,


labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#is list empty?????
def Enquiry(lis1):
    if not lis1:
        return 1
    else:
        return 0

def load_model():
    app.logger.error('Loading model')

    try:
        json_file = open('fer.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("fer.h5")
        print("Loaded model from disk")
        return loaded_model
    except FileNotFoundError as e:
        return os.abort('Unable to find the file: %s.' % str(e), 503)

def run_model(loaded_model, img):
  #  app.logger.error('Loading model')


  #  try :
   #     json_file = open('fer.json', 'r')
   #     loaded_model_json = json_file.read()
   #     json_file.close()
   #     loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
   #     loaded_model.load_weights("fer.h5")
   #     print("Loaded model from disk")
  #  except FileNotFoundError as e :
   #    return os.abort('Unable to find the file: %s.' % str(e), 503)
    pred = loaded_model.predict(img)
    prediction = labels[np.argmax(pred)]
    #print(predictio)
    return prediction

if __name__ == '__main__':
    app.run(host= '0.0.0.0', threaded=False, debug=True)