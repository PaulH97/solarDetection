import rasterio 
import numpy as np
from glob import glob
from patchify import patchify
import numpy as np
import tifffile as tiff
import os
import random
from sklearn.preprocessing import MinMaxScaler
import gdal
from tools import resampleRaster


path_data = r"D:\Universität\Master_GeoInfo\Masterarbeit\data"

os.chdir(path_data)

list_id = os.listdir(r"Sentinel1\2021\CARD_BS_MC")

for id in list_id:
    # get path of necessary data 
    print(id)
    sen1_path = glob("Sentinel1/**/*{}*_resize.tif".format(id), recursive=True)
    sen2_path = glob("Sentinel2/**/*{}*_B[2-4]*.tif".format(id), recursive=True)
    mask_path = glob("SolarParks/**/*{}*.tif".format(id), recursive=True)
    
    sen2_path.sort(reverse=True) # bands 4 3 2
    sen1_path.sort(reverse=True) # vv vh

    # all paths in one list
    sen2_sen1_mask = sen2_path + sen1_path + mask_path # [red, green, blue, vv, vh, mask] 

    print(sen2_sen1_mask)

    # define patch size 
    patch_size = 128
    scaler = MinMaxScaler()
    img_dataset = []

    print("Start patching")

    for band in sen2_sen1_mask:

        raster = rasterio.open(band)
        array = raster.read()[:,:10980,:10980]    
        array = array.reshape(10980,10980,1) # (10980,10980,1)
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
    labels = img_dataset[a*5:a*6]

    img_dataset_stack = []
    label_dataset = []

    for idx in range(len(red)):
        
        stack = np.dstack((red[idx], green[idx], blue[idx], vv[idx], vh[idx]))
        img_dataset_stack.append(stack)
        label_dataset.append(labels[idx])

    print("Finished stack")

    mask_out = os.path.join(path_data, "Crops", "All", "mask") 
    img_out = os.path.join(path_data, "Crops", "All", "img") 
    # mask_out = os.path.join(path_data, "Crops", id, "mask") 
    # img_out = os.path.join(path_data, "Crops", id, "img") 
    
    col = 0
    row = 0
    idx_noPV = []
    idx_PV = []
    for idx, label in enumerate(labels):

        # if  idx != 0 and idx %85 == 0:
        #     col = 0
        #     row += 1
        # col_row = (col, row)

        if  np.count_nonzero(label == 1):
            
            idx_PV.append(idx)

            tiff.imwrite(os.path.join(mask_out, f'{id}_mask_{idx}_pv.tif'), label)

            raster_muster = rasterio.open(sen2_sen1_mask[0])
            #transform_xy = raster_muster.transform * col_row
            #transform = rasterio.transform.from_origin(transform_xy[0], transform_xy[1], xsize=10, ysize=10)
            #crs = raster_muster.crs

            final = rasterio.open(os.path.join(img_out, f'{id}_img_{idx}_pv.tif'),'w', driver='Gtiff',
                            width=patch_size, height=patch_size,
                            count=5,
                            #crs=crs,
                            #transform=transform,
                            dtype=rasterio.float64)
        
            final.write(red[idx][:,:,0],1) # red
            final.write(green[idx][:,:,0],2) # green
            final.write(blue[idx][:,:,0],3) # blue
            final.write(vv[idx][:,:,0],4) # vv
            final.write(vh[idx][:,:,0],5) # vh
            final.close()
        
        else:
            idx_noPV.append(idx)        
        col += 1

    # random_idx = random.sample(idx_noPV,k=len(idx_PV))

    # for idx, label in enumerate(labels):

    #     if idx in random_idx:

    #         tiff.imwrite(os.path.join(mask_out, f'{id}_mask_{idx}_nopv.tif'), label)

    #         raster_muster = rasterio.open(sen2_sen1_mask[0])
    #         #transform_xy = raster_muster.transform * col_row
    #         #transform = rasterio.transform.from_origin(transform_xy[0], transform_xy[1], xsize=10, ysize=10)
    #         #crs = raster_muster.crs

    #         final = rasterio.open(os.path.join(img_out, f'{id}_img_{idx}_nopv.tif'),'w', driver='Gtiff',
    #                         width=patch_size, height=patch_size,
    #                         count=5,
    #                         #crs=crs,
    #                         #transform=transform,
    #                         dtype=rasterio.float32)
        
    #         final.write(red[idx][:,:,0],1) # red
    #         final.write(green[idx][:,:,0],2) # green
    #         final.write(blue[idx][:,:,0],3) # blue
    #         final.write(vv[idx][:,:,0],4) # vv
    #         final.write(vh[idx][:,:,0],5) # vh
    #         final.close()

