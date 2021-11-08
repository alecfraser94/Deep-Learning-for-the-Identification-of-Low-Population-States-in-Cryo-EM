%pylab inline
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from numpy import random
from numpy import expand_dims
import tensorflow as tf
from __future__ import print_function
from PIL import Image
import time
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3";
from tensorflow import keras
from keras.preprocessing.image import save_img
from keras import layers
from keras import backend as K
from keras.models import Model
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.regularizers import l2

size=128 		    #int: size of the square images
filter_size=15 		#int: size of filters in the conv2d layers 
num_filters=64		#int: number of filters in the conv2d layers
num_dense=500		#int: number of units in dense layer
dropout_rate=0.5	#float: dropout rate in between dense layers
l2_reg=0.0005		#float: L2 regulaization weight decay value in conv2d layers
lr=0.0000005		#float: learning rate for the model

starting_weights=''  			#string: location of starting weights to load into the model
data_directory='' 			    #string: location of training and validation data
viz_layer_number=0			    #int: conv2D layer number to visualize
filter_number=0			        #int: filter number to visualize
img_name=''				        #string: name of the image to visualize
    
def create_model(num_filters,filter_size,size,l2_reg,dropout_rate,num_dense,starting_weights,lr):
    """function which creates, loads in weights and compiles the convolutional neural network  
    input: num_filters, int, number of filters in the conv2d layers
    input: filter_size, int, size of filters in the conv2d layers 
    input: size, int, size of the square images
    input: l2_reg, float, L2 regulaization weight decay value in conv2d layers
    input: dropout_rate, float, dropout rate in between dense layers
    input: num_dense, int, number of units in dense layer
    input: starting_weights, string, location of starting weights to load into the model, set to empty string if starting without weights
    input: lr, float, learning rate for the model
    output: model, TF neural network model object"""
    model = Sequential()																		#create sequential model
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', input_shape=(size,size,1),padding="same",kernel_regularizer=l2(l2_reg)))		#add conv2d layer to sequential model with a relu activation layer and weight decay
    model.add(BatchNormalization())																	#add batch normalization layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))										#add max pooling layer
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', input_shape=(size/2,size/2,1),padding="same",kernel_regularizer=l2(l2_reg)))		#add conv2d layer to sequential model with a relu activation layer and weight decay
    model.add(BatchNormalization())																	#add batch normalization layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))										#add max pooling layer
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', input_shape=(size/4,size/4,1),padding="same",kernel_regularizer=l2(l2_reg)))		#add conv2d layer to sequential model with a relu activation layer and weight decay
    model.add(BatchNormalization())																	#add batch normalization layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))										#add max pooling layer
    model.add(Dropout(dropout_rate))																	#add dropout layer
    model.add(Flatten())																		#flatten
    model.add(Dense(num_dense))																		#add fully connected layer
    model.add(BatchNormalization())																	#add batch normalization layer
    model.add(Activation('relu'))																	#add relu activation layer
    model.add(Dropout(dropout_rate))																	#add dropout layer
    model.add(Dense(num_dense))																		#add fully connected layer
    model.add(BatchNormalization())																	#add batch normalization layer
    model.add(Activation('relu'))																	#add relu activation layer
    model.add(Dropout(dropout_rate))																	#add dropout layer
    model.add(Dense(2,activation='softmax'))																#add softmax layer
    if starting_weights:																		#check for weights to load
        model.load_weights(starting_weights)																#load weights if exist
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])								#compile the model with a binary_crossentropy loss
    return model																			#return model object
  
def look_at_act_maps(model,layer,filter_number,data_dir,img_name,size):
    """function which creates and plots an activation map of an image
    input: model, TF neural network model object, trained model used for the creation of activation maps
    input: layer, int, layer number corresponding to conv2d layer output which will be generated
    input: filter_number, int, 0-indexed filter number from the conv2d layer to visualize
    input: data_dir, path, location of training and validation data
    input: image_name, string, location and name of image to be made into an activation map
    input: size, int, size of the square images"""
    model2 = Model(inputs=model.inputs, outputs=model.layers[layer].output) 		#create a new model class with the same input as the trained model but which outputs the output of the specified conv2d layer
    image_path = os.path.join(data_dir, img_name)					#create the image path
    img = imread(image_path, flatten=True)						#read the image
    img = img.reshape([size, size, 1])						#reshape the image for normalization
    img=(img - img.mean())/img.std()							#normalize the image
    img = expand_dims(img, axis=0)							#expand the dimsensions of the image prior to passing it as input to the model
    pylab.imshow(img.squeeze(), cmap='gray')						#show the original image
    pylab.axis('off')									#don't show the axis labels
    pylab.show()									#display the original image
    feature_maps = model2.predict(img)						#use the new model class to predict the activation map 
    plt.imshow(feature_maps[0, :, :, filter_number], cmap='gray')			#show the activation map image
    pylab.axis('off')									#don't show the axis labels
    plt.show()										#display the activation map image
    
data_dir = os.path.abspath(data_directory)									#create path for the data directory

model=create_model(num_filters,filter_size,size,l2_reg,dropout_rate,num_dense,starting_weights,lr)		#create, compile and load weights for CNN model object 

look_at_act_maps(model,viz_layer_number,filter_number,data_dir,img_name,size)								#create activation map

