from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class CustomImageGenerator(Sequence):

    def __init__(self, X_set, y_set, output_size, augmentation=False, batch_size=8):

        self.x = X_set # Paths to all images as list
        self.y = y_set # paths to all masks as list
        self.output_size = output_size
        self.augmentation = augmentation
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):
        
        X = np.empty((self.batch_size, *self.output_size, 5)) # example shape (8,128,128,5)
        y = np.empty((self.batch_size, *self.output_size, 1))

        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        
        scaler = MinMaxScaler()

        for i, file_path in enumerate(batch_x):

            # read img as array 
            img_array = rasterio.open(file_path).read()
            img_array = np.moveaxis(img_array, 0, -1)
            img_array = scaler.fit_transform(img_array.reshape(-1, img_array.shape[-1])).reshape(img_array.shape)

            # preprocess images / data augmentation
            X[i] = img_array
            
        
        for i, file_path in enumerate(batch_y):

            # read mask as array
            mask_array = rasterio.open(file_path).read()
            mask_array = np.moveaxis(mask_array, 0, -1)
            mask_array = scaler.fit_transform(mask_array.reshape(-1, mask_array.shape[-1])).reshape(mask_array.shape)

            # preprocess mask
            y[i] = mask_array
        
        return X, y

import os
img_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\img"
mask_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\mask"

img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)

img_list.sort()
mask_list.sort()

def updatePathIMG(file_name):
    return os.path.join(img_path, file_name)

def updatePathMask(file_name):
    return os.path.join(mask_path, file_name)

img_list_update = list(map(updatePathIMG, img_list))
mask_list_update = list(map(updatePathMask, mask_list))

train_datagen = CustomImageGenerator(img_list_update, mask_list_update,(128,128))

X, y = train_datagen[1]

from matplotlib import pyplot as plt

for i in range(X.shape[0]):

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(X[i][:,:,:3])
    plt.subplot(122)
    plt.imshow(y[i])
    plt.show()

import pdb
pdb.set_trace()