
from matplotlib import pyplot as plt
import rasterio
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import os
import rasterio
from sklearn.preprocessing import MinMaxScaler
from datagen import CustomImageGenerator
import numpy as np

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

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(img_list_update, mask_list_update, test_size = 0.10, random_state = 42)

test_datagen = CustomImageGenerator(X_test, y_test, (128,128))

from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

new_model = tf.keras.models.load_model('pv_detection', compile=False, custom_objects={'dice_coef': dice_coef})

X,y = test_datagen[10]

preds_test = new_model.predict(X, verbose=1) # (len(X_test), 128, 128, 1)

preds_test_t = (preds_test > 0.5).astype(np.uint8) 

for i in range(preds_test_t.shape[0]):

    if np.count_nonzero(preds_test_t[i] == 1):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(X[i][:,:,:3])
        plt.subplot(122)
        plt.imshow(preds_test_t[i])
        plt.show()

