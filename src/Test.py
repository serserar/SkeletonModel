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
from DataGenerator import DataGenerator
import uuid
import itertools

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
    return (x_train, y_train), (x_test, y_test)  


        
def test(x_test, y_test):    
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

def saveImages(predicted_images, destinationPath):
    return

def main():
    
    print("Init")
    print("Load dataSet")
    (x_train, y_train), (x_test, y_test) = loadDataSet("../dataset/skeleton_dataset_test.tar.gz")

    print("Test")
    test(x_test)
    print("End Test")
if __name__ == '__main__':
    main()
