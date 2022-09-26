from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
import warnings
from tools import load_img_as_array

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class CustomImageGenerator(Sequence):

    def __init__(self, X_set, y_set, output_size, batch_size=8):

        self.x = X_set # Paths to all images as list
        self.y = y_set # paths to all masks as list
        self.output_size = output_size
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):
        
        X = np.empty((self.batch_size, *self.output_size, 12)) # example shape (8,128,128,12)
        y = np.empty((self.batch_size, *self.output_size, 1))

        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        
        #scaler = MinMaxScaler()

        for i, file_path in enumerate(batch_x):

            # read img as array 
            # img_array = rasterio.open(file_path).read()
            # img_array = np.moveaxis(img_array, 0, -1)
            # img_array = scaler.fit_transform(img_array.reshape(-1, img_array.shape[-1])).reshape(img_array.shape)
          
            # X[i] = img_array
            X[i] = load_img_as_array(file_path)
            
        
        for i, file_path in enumerate(batch_y):

            # read mask as array
            # mask_array = rasterio.open(file_path).read()
            # mask_array = np.moveaxis(mask_array, 0, -1)
            # mask_array = scaler.fit_transform(mask_array.reshape(-1, mask_array.shape[-1])).reshape(mask_array.shape)

            # y[i] = mask_array
            y[i] = load_img_as_array(file_path)
        
        return X, y

class CustomImageGeneratorPrediction(Sequence):

    def __init__(self, X_set, output_size, batch_size=8):

        self.x = X_set # paths to all masks as list
        self.output_size = output_size
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):
        
        X = np.empty((self.batch_size, *self.output_size, 12)) # example shape (8,128,128,12)
        
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]

        scaler = MinMaxScaler()
        
        for i, file_path in enumerate(batch_x):

            # read mask as array
            mask_array = rasterio.open(file_path).read()
            mask_array = np.moveaxis(mask_array, 0, -1)
            mask_array = scaler.fit_transform(mask_array.reshape(-1, mask_array.shape[-1])).reshape(mask_array.shape)

            # preprocess mask
            X[i] = mask_array
        
        return X