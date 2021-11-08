import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from numpy import random
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.regularizers import l2
# import all packages

size=128 		#int: size of the square images
filter_size=15 		#int: size of filters in the conv2d layers 
num_filters=64		#int: number of filters in the conv2d layers
num_dense=500		#int: number of units in dense layer
dropout_rate=0.5	#float: dropout rate in between dense layers
l2_reg=0.0005		#float: L2 regulaization weight decay value in conv2d layers
lr=0.0000005		#float: learning rate for the model

starting_weights=''  			#string: location of starting weights to load into the model, set to empty string if starting without weights
validation_data_loc=''			#string: location of validation data directory
data_directory='' 			#string: location of training and validation data
val_list='val_postive_list.txt'		#string: name and location of where to save list of stalled intermediate particle numbers for cryoSPARC
    
def load_validation_images(validate,data_dir,validation_data_loc,size):
    """function which loads validation images
    input: validate, dataframe, contains image names and labels in the validation set
    input: data_dir, path, location of training and validation data
    input: validation_data_loc, string, location of validation data directory
    input: size, int, size of the square images
    output: x_validate, np array, all validation images in a numpy array"""
    all_images = []								#array which will hold the images
    for img_name in validate.filename:						#loop through all images in the validation set
        image_path = os.path.join(data_dir,validation_data_loc, img_name)	#define image path
        img = imread(image_path, flatten=True)					#read in the image
        img = img.reshape([size, size, 1])					#reshape the image to the appropriate dimensions
        img=(img - img.mean())/img.std()					#normalize the image
        all_images.append(img)							#append the image to the array
    x_validate = np.array(all_images)						#make array into numpy array
    return x_validate								#return numpy array
    
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
    
def perform_validation(x_validate,model,val_list):
    """predict validation set images with CNN, saves a list of the predictions to be used in cryoSPARC
    input: x_validate, numpy array with validation images
    input: model, TF object, trained model to perform validation
    input: val_list, string, name of file to save list of stalled intermediate identified particles
    """
    val_pred=model.predict(x_validate)			# perform prediction on the validation dataset
    keep_list=[]					#create list which will store IDs for stalled intermediate particles
    num_rows, num_cols = val_pred.shape			#get shape of validation dataset
    val_pred_class=np.empty([num_rows,num_cols])	#create empty numpy array for binarized validation predictions
    for i in range(num_rows):				#iterate through validation dataset
        if ((val_pred[i][0] > val_pred[i][1])):		#check if contracted state is more likely
            val_pred_class[i]=[1,0]			#if so, append contracted binary class to new numpy array
        if ((val_pred[i][1] > val_pred[i][0])):		#check if stalled intermediate state is more likely
            val_pred_class[i]=[0,1]			#if so, append stalled intermediate binary class to new numpy array
            keep_list.append(i)				#append ID of current image to list containing IDs of stalled intermediates
    with open (val_list, 'w') as f:			#open file with list of stalled intermediate IDs
        for item in keep_list:				#iterate through the list
            print >> f, item				#append IDs to the list file
		
    
data_dir = os.path.abspath(data_directory)									#create path for the data directory

validate = pd.read_csv(os.path.join(data_dir,validation_data_loc, 'validate_all.csv'),sep=',')			#create dataframe of the training set from a csv file

x_validate=load_validation_images(validate,data_dir,validation_data_loc,size)					#load validation images into numpy array

model=create_model(num_filters,filter_size,size,l2_reg,dropout_rate,num_dense,starting_weights,lr)		#create, compile and load weights (optional) for CNN model object, 

perform_validation(x_validate,model,val_list)									#perform validation

