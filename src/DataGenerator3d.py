import numpy as np
import keras
import os
from os import path
import binvox_rw
from keras.callbacks import TensorBoard
from PIL import Image

class DataGenerator3d(keras.utils.Sequence):
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
        #X = X.astype('float32') / 255.
        #y = y.astype('float32') / 255.
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_list_tmp, y_list_tmp):
        'Generates data containing batch_size samples 3d' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)        
        # Generate data
        voxels = []  
        for id in x_list_tmp:
            voxel_path=os.path.join(self.dataSetPath, id)
            if os.path.exists(voxel_path):
                try:
                    with open(voxel_path, 'rb') as voxelFile:
                        voxel = np.int32(binvox_rw.read_as_3d_array(voxelFile).data)
                        voxels.append(voxel)
                except:
                    print(voxel_path)
            else:
                print(voxel_path)       
        X=np.asarray(voxels, dtype='uint8') 
         
        yvoxels=[]      
        for id in y_list_tmp:
            voxel_path=os.path.join(self.dataSetPath, id)
            if os.path.exists(voxel_path):
                try:
                    with open(voxel_path, 'rb') as voxelFile:
                        yvoxel = np.int32(binvox_rw.read_as_3d_array(voxelFile).data)
                        yvoxels.append(yvoxel)
                except:
                    print(voxel_path)
            else:
                print(voxel_path)
                       
        Y=np.asarray(yvoxels, dtype='uint8')        

        return X, Y
    