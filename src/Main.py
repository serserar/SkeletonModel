from __future__ import print_function

import keras
from keras.utils import data_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard,History
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from DataGenerator import DataGenerator
from DataGenerator3d import DataGenerator3d
import uuid
import itertools
import pickle

def downloadDatasetFromDrive(datasetId, download_dir):
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    downloaded = drive.CreateFile({'id': datasetId})
    downloaded.GetContentFile(download_dir)

def uploadFileToDrive(filePath):
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    file1 = drive.CreateFile()
    file1.SetContentFile(filePath)
    file1.Upload()

def preprocessImg(image_path):
    return Image.open(image_path)

def getXYTrain(dataSetPath, train_model):
    with open(train_model) as f:
        all_img_paths = f.read().splitlines()
    imgs = []    
    for img_path in all_img_paths:
        image_path=os.path.join(dataSetPath, img_path)
        if os.path.exists(image_path):
            try:
                imgs.append(img_to_array(Image.open(image_path)))
            except:
                print(image_path)
        else:
            print(image_path)       
    #np.empty((len(imgs), 1, 240, 320), dtype='uint8')
    XYtrain = np.asarray(imgs, dtype='uint8')     
    return XYtrain

def loadDataSet(datasetPath):
    if os.path.exists(datasetPath):
        dataset_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton')
        keras.utils.data_utils._extract_archive(datasetPath, dataset_dir, archive_format='auto')
    train_model=os.path.join(dataset_dir,"train_model")
    train_skeleton=os.path.join(dataset_dir,"train_skeleton")
    if os.path.exists(train_model):
        Xtrain = getXYTrain(dataset_dir, train_model)
    if os.path.exists(train_skeleton):
        Ytrain = getXYTrain(dataset_dir, train_skeleton)
    print("Xtrain size : " + str(len(Xtrain)))
    print("Ytrain size : " + str(len(Ytrain)))
    x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, shuffle=False, test_size=0.20)
    print("Xtr size : " + str(len(x_train)))
    print("ytr size : " + str(len(y_train)))
    print("Xtest size : " + str(len(x_test)))
    print("ytest size : " + str(len(y_test)))
    x_train = x_train.astype('float32') / 255.
    y_train = y_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_test = y_test.astype('float32') / 255.
    return (x_train, y_train), (x_test, y_test)  

def loadDataSetList2d(datasetPath):
    if os.path.exists(datasetPath):
        dataset_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton')
        keras.utils.data_utils._extract_archive(datasetPath, dataset_dir, archive_format='auto')
    train_model=os.path.join(dataset_dir,"train_model")
    train_skeleton=os.path.join(dataset_dir,"train_skeleton")
    x_train=[]
    y_train=[]
    if os.path.exists(train_model):
        with open(train_model) as f:
            all_x_img_paths = f.read().splitlines()
            for img_path in all_x_img_paths:
                x_train.append(img_path)
    if os.path.exists(train_skeleton):
        with open(train_skeleton) as f:
            all_y_img_paths = f.read().splitlines()
            for img_path in all_y_img_paths:
                y_train.append(img_path)
    print("Xtrain size : " + str(len(x_train)))
    print("Ytrain size : " + str(len(y_train)))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True, test_size=0.20)
    
    return (x_train, y_train), (x_test, y_test)  

def loadDataSetList3d(datasetPath):
    if os.path.exists(datasetPath):
        dataset_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton3d')
        keras.utils.data_utils._extract_archive(datasetPath, dataset_dir, archive_format='auto')
    train_model=os.path.join(dataset_dir,"train_model")
    train_skeleton=os.path.join(dataset_dir,"train_skeleton")
    x_train=[]
    y_train=[]
    if os.path.exists(train_model):
        with open(train_model) as f:
            all_x_img_paths = f.read().splitlines()
            for img_path in all_x_img_paths:
                x_train.append(img_path)
    if os.path.exists(train_skeleton):
        with open(train_skeleton) as f:
            all_y_img_paths = f.read().splitlines()
            for img_path in all_y_img_paths:
                y_train.append(img_path)
    print("Xtrain size : " + str(len(x_train)))
    print("Ytrain size : " + str(len(y_train)))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True, test_size=0.20)
    
    return (x_train, y_train), (x_test, y_test)  

def samp(x_train, x_test, y_train, y_test):
    training={}
    
    training_labels={}
    
    for x,y in itertools.izip(x_train,y_train) :
        id = str(uuid.uuid4())
        training[id]=x
        training_labels[id]=y
        
    validation={}
    validation_labels={}    
    for x,y in itertools.izip(x_test,y_test) :
        id = str(uuid.uuid4())
        validation[id]=x
        validation_labels[id]=y
        
    tlabel=list(training.keys())[0]
    vlabel=list(validation.keys())[0]      
    print(training[tlabel])
    print(training_labels[tlabel])
    print(validation[vlabel])
    print(validation_labels[vlabel])

## DEF A BLOCK CONV + BN + GN + MAXPOOL
def CBGN(model,filters,ishape=0):
    if (ishape!=0):
        model.add(Conv2D(filters, (3, 3), padding='same',input_shape=ishape))
    else:
        model.add(Conv2D(filters, (3, 3), padding='same'))
  
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    return model

def buildEncoder(model,filters,ishape=0):
    if (ishape!=0):
        model.add(Conv2D(filters, (3, 3), padding='same',input_shape=ishape))
    else:
        model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    return model

def buildEncoder3d(model,filters,filtersize=3,ishape=0):
    if (ishape!=0):
        model.add(Conv3D(filters, (filtersize, filtersize, filtersize), padding='same',input_shape=ishape))
    else:
        model.add(Conv3D(filters, (filtersize, filtersize, filtersize), padding='same'))
    #model.add(GN(0.3))
    model.add(BN())
    model.add(Activation('relu'))  
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    return model

def buildDecoder(model,filters):
    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(Activation('relu'))  
    model.add(BN())
    model.add(UpSampling2D((2, 2)))
    return model

def buildDecoder3d(model,filters, filtersize=3):
    model.add(Conv3D(filters, (filtersize, filtersize, filtersize), padding='same'))
    model.add(BN())
    model.add(Activation('relu'))  
    model.add(UpSampling3D(size=(2, 2, 2)))
    return model
    
def skeleton_model(input_shape):
    ## DEF NN TOPOLOGY  
    model = Sequential()

    model=buildEncoder(model,16, input_shape)
    model=buildEncoder(model,8)
    model=buildEncoder(model,8)
    model=buildDecoder(model,8)
    model=buildDecoder(model,8)
    model=buildDecoder(model,16)
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model;

def skeleton_model3d(input_shape, size):
    ## DEF NN TOPOLOGY  
    model = Sequential()
    model.add(Reshape((size, size, size, 1), input_shape=(size, size, size)))
    model=buildEncoder3d(model,64)
    model=buildEncoder3d(model,32)
    model=buildEncoder3d(model,32)
    model=buildEncoder3d(model,16)
    #model=buildEncoder3d(model,8)
    #model=buildDecoder3d(model,8)
    model=buildDecoder3d(model,16)
    model=buildDecoder3d(model,32)
    model=buildDecoder3d(model,32)
    model=buildDecoder3d(model,64)
    model.add(Conv3D(1, (3, 3, 3), activation='relu', padding='same'))
    model.add(Reshape((size, size, size), input_shape=(size, size, size, 1)))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model;

def skeleton_model3d01(input_shape):
    ## DEF NN TOPOLOGY  
    model = Sequential()
    model.add(Reshape((64, 64, 64, 1), input_shape=(64, 64, 64)))
    model=buildEncoder3d(model,64,7)
    model=buildEncoder3d(model,32,5)
    model=buildEncoder3d(model,16)
    model=buildEncoder3d(model,8)
    model=buildEncoder3d(model,8)
    model=buildDecoder3d(model,8)
    model=buildDecoder3d(model,8)
    model=buildDecoder3d(model,16)
    model=buildDecoder3d(model,32,5)
    model=buildDecoder3d(model,64,7)
    model.add(Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))
    model.add(Reshape((64, 64, 64), input_shape=(64, 64, 64, 1)))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model;

def skeleton_model3d02(input_shape):
    ## DEF NN TOPOLOGY  
    model = Sequential()
    model.add(Reshape((64, 64, 64, 1), input_shape=(64, 64, 64)))
    #model=buildEncoder3d(model,1, 64)
    model=buildEncoder3d(model,32,32)
    model=buildEncoder3d(model,16,16)
    model=buildEncoder3d(model,128,8)
    model=buildEncoder3d(model,256,4)
    model=buildDecoder3d(model,256,4)
    model=buildDecoder3d(model,128,8)
    model=buildDecoder3d(model,16,16)
    model=buildDecoder3d(model,32,32)
    #model=buildDecoder3d(model,1, 64)
    model.add(Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))
    model.add(Reshape((64, 64, 64), input_shape=(64, 64, 64, 1)))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()
    return model;

def create_model(input_shape):
    ## DEF NN TOPOLOGY  
    model = Sequential()

    model=CBGN(model,32, input_shape)
    model=CBGN(model,64)
    model=CBGN(model,128)
    model=CBGN(model,256)
    model=CBGN(model,512)

    #model.add(Flatten())
    #model.add(Dense(512))
    #model.add(Activation('relu'))

    #model.add(Dense(num_classes))
    #model.add(Activation('softmax'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    model.summary()
    return model;

def buildVGG16Model(input_shape):
    
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
    ])
    model.summary()
    return model

def segnetModel(input_shape):
    
    kernel = (3, 3)
    img_h = 240
    img_w = 320
    n_labels=2
    model = Sequential()

    encoding_layers = [Convolution2D(64, kernel, border_mode='same', input_shape=( img_h, img_w,1)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(128, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    #MaxPooling2D(),
]
    decoding_layers = [
    #UpSampling2D(),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(128, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(64, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(1, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
    model.encoding_layers = encoding_layers
    for l in model.encoding_layers:
        model.add(l)
        
    model.decoding_layers = decoding_layers
    for l in model.decoding_layers:
        model.add(l)
    #model.add(Reshape((n_labels, img_h * img_w)))
    #model.add(Permute((2, 1)))
    model.add(Activation('softmax'))
    model.summary()
    return model

def train(model, batch_size, epochs, x_train, y_train, x_test, y_test):
    

    ## OPTIM AND COMPILE
    #opt = SGD(lr=0.1, decay=1e-6)

    #model.compile(loss='categorical_crossentropy',
    #          optimizer=opt,
    #          metrics=['accuracy'])

    #model.compile(loss='binary_crossentropy',
    #          optimizer=opt,
    #          metrics=['accuracy'])

    ## TRAINING
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=[TensorBoard(log_dir='/tmp/skeletonmodel',histogram_freq=0,  write_graph=True, write_images=False)])
    
    model_path = '../test/skeletonmodel.h5'
    model.save(model_path)
    uploadFileToDrive(model_path)
    ## TEST
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def trainDataGenerator(model, batch_size, epochs, x_train, y_train, x_test, y_test):
    
    # Parameters
    params = {'dim': (240,320,1),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}
    dataSetPath = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton')
    # Generators
    training_generator = DataGenerator(dataSetPath, x_train, y_train, **params)
    validation_generator = DataGenerator(dataSetPath, x_test, y_test, **params)

    ## TRAINING
    model.fit_generator(generator=training_generator,
          validation_data=validation_generator,
          epochs=epochs, 
          verbose=1,
          shuffle=True,
          callbacks=[TensorBoard(log_dir='/tmp/skeletonmodel',histogram_freq=0,  write_graph=True, write_images=False)])
    
    model_path = '../test/skeletonmodel.h5'
    model.save(model_path)
    uploadFileToDrive(model_path)

def trainDataGenerator3d(model, batch_size, epochs, size,  x_train, y_train, x_test, y_test):
    
    # Parameters
    params = {'dim': (size, size, size, 1),
          'batch_size': batch_size,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
    history = History()
    dataSetPath = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton3d')
    # Generators
    training_generator = DataGenerator3d(dataSetPath, x_train, y_train, **params)
    validation_generator = DataGenerator3d(dataSetPath, x_test, y_test, **params)
    
    ## TRAINING
    model.fit_generator(generator=training_generator,
          validation_data=validation_generator,
          epochs=epochs, 
          verbose=1,
          shuffle=True,
          callbacks=[TensorBoard(log_dir='/tmp/skeletonmodel3d',histogram_freq=0,  write_graph=True, write_images=False), history])
    
    model_path = '../test/skeletonmodel3d_32.h5'
    model.save(model_path)
    uploadFileToDrive(model_path)
    history_path = '../test/trainHistory'
    f = open(history_path, 'wb')
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    uploadFileToDrive(history_path)
        
def test(x_test):    
    model = load_model('../test/skeletonmodel.h5')
    decoded_imgs = model.predict(x_test)
    n=10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(240, 320))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #display predict
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(240, 320))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def main():
    
    print("Init")
    batch_size = 128
    is3d=True
    continueTrain=False
    epochs = 25
    size=32
    
    if is3d:
        #https://drive.google.com/open?id=1ta01DUch2sq5qffQLJgfMBqdTv305ENe
        input_shape = (size, size, size, 1)
        downloadDatasetFromDrive("1ta01DUch2sq5qffQLJgfMBqdTv305ENe","../dataset/skeleton_3ddataset_32.tar.gz")
        print("Create model 3d")
        if continueTrain:
            model = load_model('../test/skeletonmodel3d_32.h5')
        else:    
            model = skeleton_model3d(input_shape, size) 
        
        print("Load dataSet")
        (x_train, y_train), (x_test, y_test) = loadDataSetList3d("../dataset/skeleton_3ddataset_32.tar.gz")
        print("Train 3d")
        trainDataGenerator3d(model, batch_size, epochs, size, x_train, y_train, x_test, y_test)
        print("End Train 3d")
    else:
        input_shape = (240, 320, 1)
        downloadDatasetFromDrive("1usvnmumTinLgaRIDGRJHpqNq86GGc1uF","../dataset/skeleton_dataset.tar.gz")
        print("Create model 2d")
        model = skeleton_model(input_shape)
        print("Load dataSet 2d")
        (x_train, y_train), (x_test, y_test) = loadDataSetList2d("../dataset/skeleton_dataset.tar.gz")
        print("Train 2d")
        trainDataGenerator(model, batch_size, epochs, x_train, y_train, x_test, y_test)
        print("End Train 2d")
    

    
   
    #print("Test")
    #test(x_test)
    #print("End Test")
if __name__ == '__main__':
    main()
