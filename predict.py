import rasterio
import tensorflow as tf 
import os
import numpy as np
from glob import glob
from datagen import CustomImageGeneratorPrediction
from tools import *
import yaml

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        sentinel1_folder = data['data_source']['Sentinel1']
        sentinel2_folder = data['data_source']['Sentinel2']
        model_path = data['model']['Unet_Sen1_Sen2']
        output_folder = data["output_folder"]

patching = False

sentinel1_paths = glob("{}/*.tif".format(sentinel1_folder))
sentinel2_paths = glob("{}/*.tif".format(sentinel2_folder))
sentinel_paths = sentinel1_paths + sentinel2_paths
sentinel_paths.sort() 

if patching: 

    bands_patches = {}

    for idx, band in enumerate(sentinel_paths):

        band_name = os.path.basename(band).split("_")[-1].split(".")[0]
        print("Start patching with band: ", band_name)

        raster = rasterio.open(band)
        
        if raster.transform[0] != 10:  
            raster = resampleRaster(band, 10)
            r_array = raster.ReadAsArray()
            r_array = np.expand_dims(r_array, axis=0)
        else:
            r_array = raster.read()[:,:10980,:10980]
            r_array = np.nan_to_num(r_array)
        
        r_array = np.moveaxis(r_array, 0, -1)
        
        bands_patches[band_name] = patchifyRasterAsArray(r_array, 128)

    patches_path = savePatches(bands_patches, output_folder)

patches_path = glob(r"{}/Crops/img/*.tif".format(output_folder))

patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

predict_datagen = CustomImageGeneratorPrediction(patches_path, (128,128))

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})

predictPatches(model, predict_datagen, sentinel_paths[4], output_folder)

# prediction = model.predict(img_dataset_stack, verbose=1) # output shape = (7725, 128, 128, 1)
# del img_dataset_stack
# prediction = (prediction > 0.5).astype(np.uint8) 

# prediciton_reshape = np.reshape(prediction, (85,85,1,128,128,1)) 

# recon_predict = unpatchify(prediciton_reshape, (128*85,128*85,1))

# transform = raster.transform
# crs = raster.crs

# final = rasterio.open(r'D:\Universität\Master_GeoInfo\Masterarbeit\data\prediction\test.tiff', mode='w', driver='Gtiff',
#                 width=recon_predict.shape[0], height=recon_predict.shape[1],
#                 count=1,
#                 crs=crs,
#                 transform=transform,
#                 dtype=rasterio.float32)

# final.write(recon_predict[:,:,0],1) 
# final.close()