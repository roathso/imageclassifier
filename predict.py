#!/usr/bin/env python3
'''
pip install -q -U tensorflow-gpu --user

'''

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse
import sys
import os
import json
from PIL import Image

def getflowerlabel(classes):
    flowers = []
    flower_classes = []
    for value in classes[0]:
        flower_classes.append(class_names[str(value+1)])
    flowers.append(flower_classes)
    return flowers

def predict(image_path, model, top_k = 5):
    im = Image.open(image_path)
    predict_images = np.asarray(im)
    predict_images = process_image(predict_images)
    prediction = model.predict(np.expand_dims(predict_images, axis= 0))
    
    top_k_probs, top_k_classes = tf.nn.top_k(prediction, k=top_k)
    
    probs = top_k_probs.numpy()
    classes = top_k_classes.numpy()
    
    return probs, classes

def process_image(img):
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))
    
    image = (image/255)
    return image

parser = argparse.ArgumentParser(description='Image Classification 2')
parser.add_argument('arg1',  help="path to predicted image")
parser.add_argument('arg2',  help="model class .h5")
parser.add_argument('--category_names', default = 'label_map.json', help = 'path to json class names')
parser.add_argument('--top_k', type = int, default = 5, help = 'Top predicted to print')
# Loading and checking for image and model file presence

commands = parser.parse_args()

#Check if image file is correct.
if os.path.isfile(commands.arg1) == False:
    print('Image file does not exist.')
    sys.exit()
    
#Check if model file is correct.
if os.path.isfile(commands.arg2) == False:
    print('Model file does not exist.')
    sys.exit()

# Getting additional parameters
# Checking the json file exists
try:
    with open(commands.category_names, 'r') as f:
        class_names = json.load(f)
except Exception as error:
    print(error)
    sys.exit()

    
# Loading the model
reload_model = tf.keras.models.load_model(commands.arg2
                , custom_objects = {'KerasLayer': hub.KerasLayer})


probs, classes = predict(commands.arg1, reload_model , commands.top_k)
print('Here are the top {:,} predicted'.format(commands.top_k))
#format label
class_labs=getflowerlabel(classes)
for ind in range(len(probs[0])):
    print('\nName:',class_labs[0][ind],'\nprobability:',probs[0][ind])
    