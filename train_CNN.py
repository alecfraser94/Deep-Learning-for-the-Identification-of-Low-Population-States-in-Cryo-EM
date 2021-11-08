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
num_aug=8 		#int: number of extra data augmented images to add per original image
filter_size=15 		#int: size of filters in the conv2d layers 
num_filters=64		#int: number of filters in the conv2d layers
num_dense=500		#int: number of units in dense layer
dropout_rate=0.5	#float: dropout rate in between dense layers
l2_reg=0.0005		#float: L2 regulaization weight decay value in conv2d layers
lr=0.0000005		#float: learning rate for the model
guass_mean=0.0		#float: mean of gaussian noise added to images in data augmentation
gauss_var=0.25		#float: variance of gaussian noise added to images in data augmentation
last_iteration=148	#int: the last epoch iteration, set to 0 for new training
num_iterations=50 	#int: number of epochs to run training

starting_weights=''	#string: location of starting weights to load into the model, set to empty string if starting without weights
training_data_loc=''	#string: location of training data directory
test_data_loc=''	#string: location of test data directory
data_directory=''	#string: location of training and validation data

def add_noise(image,mean,var,size):
    """function which outputs the input image with gaussian noise added
    input: image, ndarray, image to which gaussian noise will be added
    input: mean, float, mean of gaussian noise added to images in data augmentation
    input: var, float, variance of gaussian noise added to images in data augmentation
    input: size, int, size of the square images
    output: image+gauss, ndarray, image with gaussian noise added"""
    sigma=var**0.5					#calculate sigma from the variance
    gauss=np.random.normal(mean,sigma,(size,size))	#generate a ndarray of normally distributed noise with a mean and sigma as defined previously
    gauss=gauss.reshape([size, size, 1])		#reshape ndarray to be the same dimensions as input images
    return image+gauss					#return image with noise added
    
def load_training_images(train,data_dir,training_data_loc,size,guass_mean,gauss_var):
    """function which loads training images, performs data augmentation
    input: train, dataframe, contains image names and labels in the training set
    input: data_dir, path, location of training and validation data
    input: training_data_loc, string, location of training data directory
    input: size, int, size of the square images
    input: gauss_mean, float, mean of gaussian noise added to images in data augmentation
    input: gauss_var, float, variance of gaussian noise added to images in data augmentation
    output: x_train, np array, all training images in a numpy array"""
    all_images = [] 									#array which will hold the images
    for y in range(-1,2,1): 								#for loop through y direction in data augmentation
        for x in range(-1,2,1): 							#for loop through x direction in data augmentation
            data_in = tf.placeholder(tf.float32) 					#create placeholder
            data = tf.manip.roll(data_in, y, 1) 					#rolls the elements of the array; first axis is y-axis, - direction is up
            data = tf.manip.roll(data, x, 2) 						#rolls the elements of the array; second axis is x-axis, - direction is left
            for img_name in train.filename: 						#loop through all images in the training set
                image_path = os.path.join(data_dir,training_data_loc, img_name) 	#define image path
                img = imread(image_path, flatten=True) 					#read in the image
                img = img.reshape([size, size, 1]) 					#reshape the image to the appropriate dimensions
                img=(img - img.mean())/img.std()					#normalize the image
                with tf.Session() as sess:						#initiate TF graph object, close after rolling operation
                    img2=sess.run(data, {data_in: img[None, :, :, None]}) 		#perform rolling operation on image with periodic boundary conditions
                img_new=img2.reshape([size,size,1])					#reshape the image
                img_noise=add_noise(img_new,guass_mean,gauss_var,size)			#add gaussian noise to the image
                img_noise=(img_noise - img_noise.mean())/img_noise.std()		#enforce normalization on the image
                all_images.append(img_noise)						#append the image to the array
    x_train = np.array(all_images)							#make array into numpy array
    return x_train									#return numpy array
    
def load_test_images(test,data_dir,test_data_loc,size):
    """function which loads test images
    input: test, dataframe, contains image names and labels in the test set
    input: data_dir, path, location of training and validation data
    input: test_data_loc, string, location of test data directory
    input: size, int, size of the square images
    output: x_test, np array, all test images in a numpy array"""
    all_images = []							#array which will hold the images
    for img_name in test.filename:					#loop through all images in the test set
        image_path = os.path.join(data_dir,test_data_loc, img_name)	#define image path
        img = imread(image_path, flatten=True)				#read in the image
        img = img.reshape([size, size, 1])				#reshape the image to the appropriate dimensions
        img=(img - img.mean())/img.std()				#normalize the image
        all_images.append(img)						#append the image to the array
    x_test = np.array(all_images)					#make array into numpy array
    return x_test							#return numpy array
    
def create_augmented_training_label_array(y_train,num_aug):
    """function which creates an augmented training label array from the original non augmented training label array
    input: y_train, binary matrix, training labels
    input: num_aug, int, number of extra data augmented images to add per original image
    output: y_train, numpy array, labels for data augmented training set"""
    length,width=y_train.shape			#get length of the original label matrix
    y_train_large=[]				#create new array
    for j in range(num_aug+1):			#loop through new data augmentation multiple of images
        for i in range(length):			#loop through original training labels
            y_train_large.append(y_train[i])	#append label to larger data augmented array
    y_train=np.array(y_train_large)		#turn array into numpy array
    return y_train				#return numpy array
    
def create_model(num_filters,filter_size,size,l2_reg,dropout_rate,num_dense,starting_weights,lr):
    """function which creates, load weights into and compiles convolutional neural network  
    input: num_filters, int, number of filters in the conv2d layers
    input: filter_size, int, size of filters in the conv2d layers 
    input: size, int, size of the square images
    input: l2_reg, float, L2 regulaization weight decay value in conv2d layers
    input: dropout_rate, float, dropout rate in between dense layers
    input: num_dense, int, number of units in dense layer
    input: starting_weights, string, location of starting weights to load into the model, set to '' if starting without weights
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
        model.load_weights(starting_weights)																#load weights if they exist
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])								#compile the model with a binary_crossentropy loss
    return model																			#return model object
    
def train_network(model,num_iterations,x_train,y_train,x_test,y_test,last_iteration):
    """function which trains the neural network with training data, then save weights after each epoch and tests the metrics after each epoch
    input: model, TF object, cnn model to train
    input: num_iterations, int, number of epochs to run training
    input: x_train, np array, training images
    input: y_train, np array, training labels
    input: x_test, np array, test images
    input: y_test, np array, test labels
    input: last_iteration, int, the last epoch iteration, set to 0 for new training"""
    for j in range(num_iterations):											#iterate through training iterations
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1,shuffle=True)				#perform training for one epoch
        model.save("weights_at_epoch_%d.h5" % (j+last_iteration+1))							#save model weights after training epoch
        y_pred =model.predict(x_test)											#make predictions on the test set
        num_rows, num_cols = y_pred.shape										#get shape of the test set
        y_pred_class=np.empty([num_rows,num_cols])									#create new np array for binarized predictions
        for i in range(num_rows):											#iterate through test set
            if (y_pred[i][0] > y_pred[i][1]):										#check if probabilty of contracted image is higher than stalled intermediate
                y_pred_class[i]=[1,0]											#set binary result to contracted
            if (y_pred[i][1] > y_pred[i][0]):										#check if probabilty of stalled intermediate image is higher than contrated
                y_pred_class[i]=[0,1]											#set binary result to stalled intermediate
        print(classification_report(y_test, y_pred_class,target_names=['contracted', 'stalled intermediate']))		#print classification report on the test data
        print("done epoch %d" % (j+last_iteration+1))									#print end of iteration
    
    
data_dir = os.path.abspath(data_directory)									#create path for the data directory

train = pd.read_csv(os.path.join(data_dir,training_data_loc, 'train.csv'),sep=',')				#create dataframe of the training set from a csv file
test = pd.read_csv(os.path.join(data_dir,test_data_loc, 'test.csv'),sep=',')					#create dataframe of the test set from a csv file

y_train=to_categorical(train.label)										#create binary matrix from dataframe for training labels
y_test=to_categorical(test.label)										#create binary matrix from dataframe for test labels

y_train=create_augmented_training_label_array(y_train,num_aug)							#create new numpy array for expanded, data augmented training set

x_train=load_training_images(train,data_dir,training_data_loc,size,guass_mean,gauss_var)			#load training images into numpy array for training

x_test=load_test_images(test,data_dir,test_data_loc,size)							#load test images into numpy array for validation

model=create_model(num_filters,filter_size,size,l2_reg,dropout_rate,num_dense,starting_weights,lr)		#create, compile and load weights (optional) for CNN model object, 

train_network(model,num_iterations,x_train,y_train,x_test,y_test,last_iteration)				#train network, create validation reports and save model weights

