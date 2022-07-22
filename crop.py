import os
import rasterio 
import numpy as np
from glob import glob
from patchify import patchify
import tifffile as tiff

# load sentinel 1 & 2 band paths as list
sen1_path_resize = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\S1_CARD-BS-MC_202110_33UVT_*_resize.tif")
sen2_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\SENTINEL2X_20210915-000000-000_L3A_T33UVT_C_V1-2_FRC_B*.tif")
sen2_path.sort(reverse=True) # bands 4 3 2

# mask path as list
mask = [r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT\pv_anlagen.tif"]

# alle paths in one list
sen2_sen1_mask = sen2_path + sen1_path_resize + mask # [red, green, blue, vv, vh, mask]

# define patch size 
patch_size = 128

img_dataset = []

for band in sen2_sen1_mask:

    raster = rasterio.open(band)
    array = raster.read().reshape(10980,10980,1)

    # resize array for patch size? 
    patches = patchify(array, (patch_size, patch_size, 1), step=patch_size)

    patchX = patches.shape[0]
    patchY = patches.shape[1]

    for i in range(patchX):
        for j in range(patchY):
           
            single_patch_img = patches[i,j,:,:]
            # single_patch_img = (single_patch_img.astype('float32')) / 255. 
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

#Another Sanity check, view few mages
import random
import numpy as np
from matplotlib import pyplot as plt

idx_list = []

for idx, label in enumerate(labels):
    if  np.count_nonzero(label == 1):
        idx_list.append(idx)


image_number = random.randint(0, len(img_dataset_stack))
image_number = random.choice(idx_list)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_dataset_stack[image_number][:,:,0])
plt.subplot(122)
plt.imshow(label_dataset[image_number])
plt.show()

import pdb
pdb.set_trace()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)
































# mask_out = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT\crops"

# for idx, label in enumerate(labels):
#     if  np.count_nonzero(label == 1):
#         print(idx)
#     tiff.imwrite(os.path.join(mask_out, f'33UVT_label_{idx}.tif'), label)




# raster_muster = rasterio.open(sen2_sen1_mask[0])

# for idx in range(len(red)):

#     final = rasterio.open(os.path.join(out_path, f'33UVT_RGB_VV_VH_{idx}.tif'),'w', driver='Gtiff',
#                         width=patch_size, height=patch_size,
#                         count=5,
#                         crs=raster_muster.crs,
#                         transform=raster_muster.transform,
#                         dtype=raster_muster.dtypes[0]
#                         )
       
#     final.write(red[idx][:,:,0],1) # red
#     final.write(green[idx][:,:,0],2) # green
#     final.write(blue[idx][:,:,0],3) # blue
#     final.write(vv[idx][:,:,0],4) # vv
#     final.write(vh[idx][:,:,0],5) # vh