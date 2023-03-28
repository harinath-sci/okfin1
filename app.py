
from __future__ import division, print_function
from flask import Flask, url_for, request, render_template


import os


import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

global model,graph
import tensorflow as tf
app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')
model = load_model('models/fruit_predict.h5')
@app.route('/', methods=['POST'])
def predict():
    lk=request.files['fl']
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(
            basepath, 'uploads', secure_filename(lk.filename))
    lk.save(image_path)
    preds=model_predict(image_path,model)
    label = np.argmax(preds)
    ls=["Apple","Banana","Bread","Egg","Lemon","Milk","Onion","Orange","Potato","Raw Mango","Rice","Tomato","Wheat","White Onion"]
    result = ls[label]
    if(result=="Apple"):
        c=52
        p=0.3
        tf=0.2
        tc=14
    elif(result=="Banana"):
        c=89
        p=1.1
        tf=0.3
        tc=23

    elif(result=="Raw Mango"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Bread"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Egg"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Lemon"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Milk"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Onion"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Orange"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Potato"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Wheat"):
        c=60
        p=0.8
        tf=0.4
        tc=15

    elif(result=="Rice"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    elif(result=="Tomato"):
        c=60
        p=0.8
        tf=0.4
        tc=15
    else:
        c=60
        p=0.8
        tf=0.4
        tc=15
    li=[]
    s="Fruit Name : "+result
    s1="Calorie count : "+str(c)
    s2="Protein count : "+str(p)
    s3="Total Fat : "+str(tf)
    s4="Total Carbohydratyes : "+str(tc)
    li.append(s)
    li.append(s1)
    li.append(s2)
    li.append(s3)
    li.append(s4)
    return render_template('index.html',prediction=li)
    
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(64, 64))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

if __name__ == '__main__':
    app.run(port=3000,debug=True)