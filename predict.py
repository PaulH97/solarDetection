
from weakref import finalize
from xml.etree.ElementPath import prepare_descendant
from matplotlib import pyplot as plt
import rasterio
import tensorflow as tf 
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras import backend as K
from patchify import patchify, unpatchify
from glob import glob

sen2_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\32UPU\SENTINEL2X_20210815-000000-000_L3A_T32UPU_C_V1-2_FRC_B*.tif")
sen1_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\2021\CARD_BS_MC\32UPU\S1_CARD-BS-MC_202108_32UPU_V*_resize.tif")

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

del red
del green
del blue
del vv
del vh
del img_dataset
del stack

print("Finished stack")

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

new_model = tf.keras.models.load_model('pv_detection', compile=False, custom_objects={'dice_coef': dice_coef})

preds_test = new_model.predict(img_dataset_stack, verbose=1) # output shape = (7725, 128, 128, 1)
del img_dataset_stack
preds_test_t = (preds_test > 0.5).astype(np.uint8) 

preds_test_t_reshape = np.reshape(preds_test_t, (patches.shape)) #(85,85,1,128,128,1)

recon = unpatchify(preds_test_t_reshape, (patch_size*patchX,patch_size*patchY,1))

transform = raster.transform
crs = raster.crs

final = rasterio.open(r'D:\Universität\Master_GeoInfo\Masterarbeit\data\prediction\test.tiff', mode='w', driver='Gtiff',
                width=recon.shape[0], height=recon.shape[1],
                count=1,
                crs=crs,
                transform=transform,
                dtype=rasterio.float32)

final.write(recon[:,:,0],1) 
final.close()