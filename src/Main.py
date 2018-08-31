from __future__ import print_function

import keras
from keras.utils import data_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

def downloadDatasetFromDrive(datasetId, download_dir):
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    downloaded = drive.CreateFile({'id': datasetId})
    downloaded.GetContentFile(download_dir)

def download_tracking_file_by_id(file_id, download_dir):
    gauth = GoogleAuth(settings_file='../settings.yaml')
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("../credentials.json")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("../credentials.json")

    drive = GoogleDrive(gauth)

    file6 = drive.CreateFile({'id': file_id})
    file6.GetContentFile(download_dir+'mapmob.zip')

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
    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    return (x_train, y_train), (x_test, y_test)
    #x_train = np.empty((num_train_samples, 3, 320, 240), dtype='uint8')
    #y_train = np.empty((num_train_samples,), dtype='uint8')    

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

def buildDecoder(model,filters):
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
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
    
    model.save('../test/skeletonmodel.h5')
    ## TEST
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    #decoded_imgs = model.predict(x_test)
        
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
    batch_size = 25

    epochs = 25
    #https://drive.google.com/open?id=1N0PHqTlM7zWkg_q8LizpRCIJQLft1bIZ
    downloadDatasetFromDrive("1N0PHqTlM7zWkg_q8LizpRCIJQLft1bIZ","../dataset/skeleton_dataset.tar.gz")
    
    input_shape = (240, 320, 1)
    print("Create model")
    model = skeleton_model(input_shape)
    print("Load dataSet")
    (x_train, y_train), (x_test, y_test) = loadDataSet("../dataset/skeleton_dataset.tar.gz")
    input_shape = x_train.shape[1:]
    
    print("Train")
    train(model, batch_size, epochs, x_train, y_train, x_test, y_test)
    print("End Train")
    #print("Test")
    #test(x_test)
    #print("End Test")
if __name__ == '__main__':
    main()
