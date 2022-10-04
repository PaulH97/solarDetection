from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler
import warnings
from tools import load_img_as_array

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class CustomImageGenerator(Sequence):

    def __init__(self, X_set, y_set, output_size, bands, batch_size=16):

        self.x = X_set # Paths to all images as list
        self.y = y_set # paths to all masks as list
        self.output_size = output_size
        self.band_count = bands
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):

        X = np.empty((self.batch_size, *self.output_size, self.band_count)) # example shape (8,128,128,12)
        y = np.empty((self.batch_size, *self.output_size, 1))

        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        for i, file_path in enumerate(batch_x):

            X[i] = load_img_as_array(file_path)
            
        
        for i, file_path in enumerate(batch_y):

            y[i] = load_img_as_array(file_path)
        
        return X, y

class CustomImageGeneratorPrediction(Sequence):

    def __init__(self, X_set, output_size, bands, batch_size=5):

        self.x = X_set # paths to all imgages as list
        self.output_size = output_size
        self.band_count = bands
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):
        
        X = np.empty((self.batch_size, *self.output_size, self.band_count)) # example shape (5,128,128,12)
        
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
               
        for i, file_path in enumerate(batch_x):

            X[i] = load_img_as_array(file_path)
        
        return X