import numpy as np
import keras
import os
from os import path
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from PIL import Image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataSetPath, x_list, y_list, batch_size=32, dim=(240,320), n_channels=1,
                 n_classes=10, shuffle=True ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.x_list = x_list
        self.y_list = y_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataSetPath=dataSetPath
        
    def __genListIDs__(self, train_model, train_skeleton):
        return
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_list_tmp = [self.x_list[k] for k in indexes]
        y_list_tmp = [self.y_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(x_list_tmp, y_list_tmp)
        X = X.astype('float32') / 255.
        y = y.astype('float32') / 255.
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_list_tmp, y_list_tmp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)        
        # Generate data
        imgs = []  
        for id in x_list_tmp:
            image_path=os.path.join(self.dataSetPath, id)
            if os.path.exists(image_path):
                try:
                    imgs.append(img_to_array(Image.open(image_path)))
                except:
                    print(image_path)
            else:
                print(image_path)       
        X=np.asarray(imgs, dtype='uint8') 
         
        yimgs=[]      
        for id in y_list_tmp:
            image_path=os.path.join(self.dataSetPath, id)
            if os.path.exists(image_path):
                try:
                    yimgs.append(img_to_array(Image.open(image_path)))
                except:
                    print(image_path)
            else:
                print(image_path)
                       
        Y=np.asarray(yimgs, dtype='uint8')        

        return X, Y
    