# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:34:28 2022
@author: FrankNGUEN
# https://www.youtube.com/watch?v=C42lHmnNFe8&t=4s
"""
#------------------------------------------------------------------------------
# import thu vien
#------------------------------------------------------------------------------
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt                                
#------------------------------------------------------------------------------
# Ham xu ly chuan hoa anh
#------------------------------------------------------------------------------
## ham goi VGG16 model
def get_extract_model():
    vgg16_model   = VGG16(weights="imagenet")                  #VGG16 co san trong keras
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output) #VGG16 model architecture --> extract layer fc1    
    return extract_model
## Ham chuyen doi hinh anh thanh tensor
def image_preprocess(img):                                     ## Ham tien xu ly hinh anh
    img = img.resize((224,224))                                #VGG16 dung 224, RGB --> lam dung dau vao
    img = img.convert("RGB")                                   #VGG16 dung 224, RGB --> lam dung dau vao
    x   = image.img_to_array(img)                              #Chuyen imgage thanh mang
    x   = np.expand_dims(x, axis=0)                            #Them 1 chieu de thanh tensor
    x   = preprocess_input(x)                                  #Tien xu ly theo VGG16
    return x
## Ham trich xuat dac trung anh
def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img        = Image.open(image_path)                        #Mo anh
    img_tensor = image_preprocess(img)                         #Su dung ham tien xu ly hinh anh
    vector     = model.predict(img_tensor)[0]                  #Trich dac trung, lay phan tu 0
    vector     = vector / np.linalg.norm(vector)               #Chuan hoa vector = chia cho L2 norm
    return vector
#------------------------------------------------------------------------------
#load model
#------------------------------------------------------------------------------
search_image  = "datatest/lion.jpeg"                           # Dinh nghia anh can tim kiem
model         = get_extract_model()                            # Khoi tao model
search_vector = extract_vector(model, search_image)            # Trich dac trung anh search
vectors       = pickle.load(open("model/vectors.pkl","rb"))    # Load 4700 vector tu vectors.pkl ra bien
paths         = pickle.load(open("model/paths.pkl","rb"))      # Load 4700 vector tu paths.pkl ra bien
distance      = np.linalg.norm(vectors - search_vector, axis=1)# Tinh khoang cach tu search_vector den tat ca cac vector
K             = 16                                             # Sap xep va lay ra K vector co khoang cach ngan nhat
ids           = np.argsort(distance)[:K]
nearest_image = [(paths[id], distance[id]) for id in ids]      # Tao oputput
#------------------------------------------------------------------------------
# Ve len man hinh cac anh gan nhat do
#------------------------------------------------------------------------------
axes          = []
grid_size     = int(math.sqrt(K))
fig           = plt.figure(figsize=(10,5))
for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))
    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))
fig.tight_layout()
plt.show()
#------------------------------------------------------------------------------