# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:34:28 2022
@author: FrankNGUEN
# https://www.youtube.com/watch?v=C42lHmnNFe8&t=4s
"""
#------------------------------------------------------------------------------
# import thu vien
#------------------------------------------------------------------------------
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model
from PIL import Image
import pickle
import numpy as np
#------------------------------------------------------------------------------
# Ham tao model
#------------------------------------------------------------------------------
def get_extract_model():
    vgg16_model   = VGG16(weights="imagenet")                  #VGG16 co san trong keras
    extract_model = Model(inputs=vgg16_model.inputs, 
                          outputs=vgg16_model.get_layer("fc1").output) #VGG16 model architecture --> extract layer fc1    
    return extract_model
# Ham chuyen doi hinh anh thanh tensor
## Ham tien xu ly hinh anh
def image_preprocess(img):
    img = img.resize((224,224))                                #VGG16 dung 224, RGB --> lam dung dau vao
    img = img.convert("RGB")                                   #VGG16 dung 224, RGB --> lam dung dau vao
    x   = image.img_to_array(img)                              #Chuyen img thanh mang
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
data_folder = "dataset"                                        #Dinh nghia thu muc data
model       = get_extract_model()                              #Khoi tao model
vectors     = []                                               #Khai bao 1 list vectors
paths       = []                                               #Khai bao 1 list paths
for image_path in os.listdir(data_folder):
    image_path_full = os.path.join(data_folder, image_path)    #Noi full path
    image_vector    = extract_vector(model,image_path_full)    #Trich dac trung
    vectors.append(image_vector)                               #Add dac trung vao list                  
    paths.append(image_path_full)                              #Add full path vao list
vector_file = "model/vectors.pkl"                              #save vao file     
path_file   = "model/paths.pkl"                                #save vao file
#------------------------------------------------------------------------------
pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))
#------------------------------------------------------------------------------