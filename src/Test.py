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
from keras.preprocessing.image import img_to_array,array_to_img
from keras.callbacks import TensorBoard
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os import path
from os.path import basename
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from DataGenerator import DataGenerator
from DataGenerator3d import DataGenerator3d
import uuid
import itertools
import binvox_rw
import tarfile
import pickle

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
    dataset_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton')
    if os.path.exists(datasetPath):
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

def loadDataSetList3d(datasetPath):
    if os.path.exists(datasetPath):
        dataset_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton3dtest')
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
        
def test(x_test, y_test):    
    model = load_model('../test/skeletonmodel_final_2.h5')
    decoded_imgs = model.predict(x_test)
    saveImages(decoded_imgs, "../predicted")
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
    
def test3d(x_test, y_test):    
    model = load_model('../test/skeletonmodel3d_32.h5')
    #plotResults(model)
    dataSetPath = os.path.join(os.path.expanduser('~'), '.keras/datasets/skeleton3dtest')
    params = {'dim': (64, 64, 64, 1),
          'batch_size': 16,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}
    # Generators
    validation_generator = DataGenerator3d(dataSetPath, x_test, y_test, **params)
    #scoreSeg = model.evaluate_generator(validation_generator, len(x_test))
    #print("Accuracy = ",scoreSeg[1])
    #predicted_voxels = model.predict_generator(validation_generator, len(x_test))
    predicted_dir = "../predicted"
    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)
    
    archive_name = "../test/preficted.tar.gz"    
    tar = tarfile.open(archive_name, "w:gz")    
    for i in range(len(x_test)):
        voxels = []
        voxel_path=os.path.join(dataSetPath, x_test[i])
        if os.path.exists(voxel_path):
                try:
                    with open(voxel_path, 'rb') as voxelFile:
                        voxel = binvox_rw.read_as_3d_array(voxelFile).data.astype(np.float32)
                        voxels.append(voxel)
                except:
                    print(voxel_path)
        else:
            print(voxel_path)
              
        Xtest=np.asarray(voxels, dtype='float32')         
        predicted_voxels = model.predict(Xtest, batch_size=1)
        #print("predict : " + str(i))
        filename = basename(x_test[i])
        filename = os.path.splitext(filename)[0]
        saveVoxels(predicted_voxels, predicted_dir, filename , tar)
        print("saved : " + filename)
        print("predict : ")
        yvoxels = []
        yvoxel_path=os.path.join(dataSetPath, x_test[i])
        if os.path.exists(yvoxel_path):
                try:
                    with open(yvoxel_path, 'rb') as yvoxelFile:
                        yvoxel = binvox_rw.read_as_3d_array(yvoxelFile).data.astype(np.float32)
                        yvoxels.append(yvoxel)
                except:
                    print(yvoxel_path)
        else:
            print(yvoxel_path)
              
        Ytest=np.asarray(yvoxels, dtype='float32')         
        score = model.evaluate(Xtest, Ytest, verbose=0)
        print('Test loss:', score)
        
    tar.close()    
    uploadFileToDrive(archive_name)

    
def saveImages(predicted_images, destinationPath):
    for image_array in predicted_images:
        img = array_to_img(image_array)
        id = str(uuid.uuid4())
        imgpath = os.path.join(destinationPath, id + ".ppm")
        img.save(imgpath,'ppm')
    return

def saveVoxels(predicted_voxels, destinationPath, name, tar):
    for voxel_array in predicted_voxels:
        voxel_path = os.path.join(destinationPath, name + ".binvox")
        voxel = binvox_rw.Voxels(voxel_array > 0.5, voxel_array.shape, (0, 0, 0), 1,'xyz')
        with open(voxel_path, 'wb') as f:
            voxel.write(f)
            
        np_path = os.path.join(destinationPath, name + ".npy")    
        np.save(np_path,  voxel_array)
        tar.add(voxel_path, name + ".binvox")
        tar.add(np_path, name + ".npy") 
        
    
def plotResults(model):
    epochs=5
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
def plot_history(history):
    loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history[l], 'b', label='Training loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()        

def uploadFileToDrive(filePath):
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    file1 = drive.CreateFile()
    file1.SetContentFile(filePath)
    file1.Upload()

def main():
    
    is3d=True
    is2d=False
        
    if is3d:
        print("Init 3d")
        print("Load 3d dataSet")
        #https://drive.google.com/open?id=1UsGkRzUKyK_AsTsAa1BfZn06RK8n86zS
        #downloadDatasetFromDrive("1UsGkRzUKyK_AsTsAa1BfZn06RK8n86zS","../dataset/skeleton_3ddataset32test.tar.gz")
        (x_train, y_train), (x_test, y_test) = loadDataSetList3d("../dataset/skeleton_3ddataset32test.tar.gz")
        print("Test 3d")
        test3d(x_test, y_test)
        print("End Test 3d")
    elif is2d:    
        print("Init 2d")
        print("Load 2d dataSet")
        (x_train, y_train), (x_test, y_test) = loadDataSet("../dataset/skeleton_dataset_test.tar.gz")
        print("Test 2d")
        test(x_test, y_test)
        print("End Test 2d")
    

        
if __name__ == '__main__':
    main()
