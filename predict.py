
from matplotlib import pyplot as plt
import rasterio
import tensorflow as tf 
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras import backend as K
from patchify import patchify
from glob import glob

sen2_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\32UNA\SENTINEL2X_20210215-000000-000_L3A_T32UNA_C_V1-2_FRC_B*.tif")
sen1_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\2021\CARD_BS_MC\32UNA\S1_CARD-BS-MC_202102_32UNA_V*_resize.tif")

sen2_path.sort(reverse=True) # bands 4 3 2
sen1_path.sort(reverse=True) # vv vh

sen2_sen1 = sen2_path + sen1_path

patch_size = 128
scaler = MinMaxScaler()

img_dataset = []

for band in sen2_sen1:

    raster = rasterio.open(band)
    array = raster.read()[:,:10980,:10980]    
    array = np.moveaxis(array, 0, -1)
    # resize array for patch size? 
    patches = patchify(array, (patch_size, patch_size, 1), step=patch_size)

    patchX = patches.shape[0]
    patchY = patches.shape[1]

    for i in range(patchX):
        for j in range(patchY):
        
            single_patch_img = patches[i,j,:,:]
            # need to normalize values between 0-1
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
            img_dataset.append(single_patch_img)

a = patchX*patchY

red = img_dataset[0:a]
green = img_dataset[a:a*2]
blue = img_dataset[a*2:a*3]
vv = img_dataset[a*3:a*4]
vh = img_dataset[a*4:a*5]

patches_count = len(red)

img_dataset_stack = np.empty((patches_count, patch_size, patch_size, 5)) 

for idx in range(patches_count):
    
    stack = np.dstack((red[idx], green[idx], blue[idx], vv[idx], vh[idx]))
    img_dataset_stack[idx] = stack

print("Finished stack")

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

new_model = tf.keras.models.load_model('pv_detection', compile=False, custom_objects={'dice_coef': dice_coef})

preds_test = new_model.predict(img_dataset_stack, verbose=1) # (len(X_test), 128, 128, 1)
preds_test_t = (preds_test > 0.5).astype(np.uint8) 

for i in range(preds_test_t.shape[0]):

    if np.count_nonzero(preds_test_t[i] == 1):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(img_dataset_stack[i][:,:,:3])
        plt.subplot(122)
        plt.imshow(preds_test_t[i])
        plt.show()

import pdb
pdb.set_trace()
