# Flask utils
import re
import os
import cv2
import pickle
import string
import tensorflow
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from os import listdir
import matplotlib.pyplot as plt
import efficientnet.keras as effnet
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

inputvideopath = './static/images/uploadedvideo/input_video.mp4'
outputvideopath = './static/images/uploadedvideo/output_video.mp4'

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
default_image_size = tuple((128, 128))
image_size = 128
model =load_model('./model/EfficientnetB2.h5')

class_labels = pd.read_pickle('label_transform.pkl')
classes = class_labels.classes_

#flask route
@app.route('/')
def index(): 
  return render_template('index.html')

@app.route('/home')
def home(): 
  return render_template('index.html')

@app.route('/service')
def service():
  return render_template('service.html',outputvalues='nonoutput')


@app.route('/weapondetection',methods=['POST'])
def weapondetection():
  if os.path.exists(inputvideopath):
      os.remove(inputvideopath)
  if os.path.exists(outputvideopath):
      os.remove(outputvideopath)
  file = None
  file = request.files['file']
  file.save(inputvideopath)

  vs = cv2.VideoCapture(inputvideopath)
  writer = None
  (W, H) = (None, None)
  writer = None
  # loop over the video 
  while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    # frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    

    output = frame.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = np.array([image])
    #prediction
    prediction=model.predict(image)
    pred_= prediction[0]
    pred=[]
    for ele in pred_:
      pred.append(ele)
    maxi_ele = max(pred)
    idx = pred.index(maxi_ele)
    final_class=classes
    class_name= final_class[idx]
    

    # write on the output frame
    newsize = (256, 256)
    text = "weapon detected: {}".format(class_name)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (0, 255, 0), 5)
    
    if writer is None:
            # initialize our video writer
            writer = cv2.VideoWriter(outputvideopath,cv2.VideoWriter_fourcc(*'H264'), 26, (frame.shape[1], frame.shape[0]),True)
            #writer = cv2.VideoWriter(outputvideopath, fourcc, 1,(frame.shape[1], frame.shape[0]),True)

    # write the output 
    writer.write(output)

  return render_template('service.html', videopath=outputvideopath)
  



if __name__ == '__main__':
    app.run(debug=False)



