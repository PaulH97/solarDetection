
import rasterio 
import numpy as np
from glob import glob
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import os

# load sentinel 1 & 2 band paths as list
sen1_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\2021\CARD_BS_MC\33UVT\S1_CARD-BS-MC_202110_33UVT_*.tif")
sen2_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\SENTINEL2X_20210915-000000-000_L3A_T33UVT_C_V1-2_FRC_B*.tif")
sen2_path.sort(reverse=True) # bands 4 3 2
sen1_path.sort(reverse=True) # vv vh

# mask path as list
mask = [r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT\pv_anlagen.tif"]

# alle paths in one list
sen2_sen1_mask = sen2_path + sen1_path + mask # [red, green, blue, vv, vh, mask]

# define patch size 
patch_size = 128
img_dataset = []

for band in sen2_sen1_mask:

    print("start with band: ", band)

    raster = rasterio.open(band)
    array = raster.read()[:,:10980,:10980]    
    array = array.reshape(10980,10980,1) # (10980,10980,1)
    print(array.shape)
    # resize array for patch size? 
    patches = patchify(array, (patch_size, patch_size, 1), step=patch_size)

    patchX = patches.shape[0]
    patchY = patches.shape[1]

    for i in range(patchX):
        for j in range(patchY):
           
            single_patch_img = patches[i,j,:,:]
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
            img_dataset.append(single_patch_img)

a = patchX*patchY

red = img_dataset[0:a]
green = img_dataset[a:a*2]
blue = img_dataset[a*2:a*3]
vv = img_dataset[a*3:a*4]
vh = img_dataset[a*4:a*5]
labels = img_dataset[a*5:a*6]

img_dataset_stack = []
label_dataset = []

for idx in range(len(red)):
    
    stack = np.dstack((red[idx], green[idx], blue[idx], vv[idx], vh[idx]))
    img_dataset_stack.append(stack)
    label_dataset.append(labels[idx])

idx_list = []

for idx, label in enumerate(labels):
    if  np.count_nonzero(label == 1):
        idx_list.append(idx)

import pdb
pdb.set_trace()

mask_out = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\crops\mask"
img_out = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\crops\img"

for idx, label in enumerate(labels):
    if  np.count_nonzero(label == 1):

        tiff.imwrite(os.path.join(mask_out, f'mask_{idx}.tif'), label)

        raster_muster = rasterio.open(sen2_sen1_mask[0])

        final = rasterio.open(os.path.join(img_out, f'img_{idx}.tif'),'w', driver='Gtiff',
                        width=patch_size, height=patch_size,
                        count=5,
                        dtype=rasterio.float32)
       
        final.write(red[idx][:,:,0],1) # red
        final.write(green[idx][:,:,0],2) # green
        final.write(blue[idx][:,:,0],3) # blue
        final.write(vv[idx][:,:,0],4) # vv
        final.write(vh[idx][:,:,0],5) # vh

