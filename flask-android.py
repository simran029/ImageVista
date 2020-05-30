from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import urllib.request
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
from gtts import gTTS
global k,i
language='en'

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

file = 'data\coco_classes.txt'

yolo = YOLO(0.6, 0.5)
def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names
	
	
all_classes = get_classes(file)
def process_image(img):
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image

def detect_image(image, yolo, all_classes):

    pimage = process_image(image)
    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:

        #draw(image, boxes, scores, classes, all_classes)
        print(classes)
        #classes = classes[0].strip(' ')
        print(len(classes))
        mytext = ""
        k=[]
        for i in range(len(classes)):
            print(all_classes[classes[i]])
            if all_classes[classes[i]] not in k:
                mytext += " "+all_classes[classes[i]]
                k.append(all_classes[classes[i]])
        mytext += "present in the scene." 
        myobj = gTTS(text=mytext, lang=language, slow=True)  
        myobj.save(str(i)+".mp3") 
        os.system(str(i)+".mp3")
		#os.remove(str(i)+".mp3")
        return mytext
    return ""



	
def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im
	
app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    f = request.files['file']

        # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'images\\res\\', secure_filename(f.filename))
    f.save(file_path)
    image = cv2.imread(file_path)
    result = detect_image(image, yolo, all_classes)
                #cv2.imwrite('images\\res\\' + f, image)           # Convert to string
    return result


app.run(host="0.0.0.0", port=5000, debug=True)