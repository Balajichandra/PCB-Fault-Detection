import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#To allocate memorey for gpu(here we are using 50%)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)

def get_model():
    global model
    model = load_model('model.h5')
    print("Model loaded!")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    print(pred)
    
    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0
    
    print(labels)
    final=np.array(labels)
    
    if final[0][0]==1:
        return "PCB is Bad"
    else:
        return "PCB is Good"
get_model()
@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'C:/Users/Balaji/Documents/Machine Learning/defect de/Defect-Detection-of-PCB-main/test', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)            #file_path can or may used at the place of filename                
if __name__ == "__main__":
    app.run()    