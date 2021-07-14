import numpy as np
import time
import os
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

batch_size = 32
file_name = 'vgg16_sign_fine_casual_2'

classes = ['casual_egao', 'casual_magao','formal_egao','formal_magao']

modeljson = open(file_name + '.json').read()
model_vgg = model_from_json(modeljson)
model_vgg.load_weights(file_name+'.h5')

filename = './disp/amarec(20210707-1420).avi_380_19.00.jpg'

car_vgg = image.load_img(filename, target_size=(224, 224))

vgg = image.img_to_array(car_vgg)
vgg = np.expand_dims(vgg, axis=0)
vgg = preprocess_input(vgg)

pred_vgg = model_vgg.predict(vgg)[0]
predicted = pred_vgg.argmax()
text_vgg = '[No.' + str(predicted) + '] '  + classes[predicted]
print(text_vgg)