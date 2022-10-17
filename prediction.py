import rasterio
import tensorflow as tf 
import os
import numpy as np
from glob import glob
from datagen import CustomImageGeneratorPrediction
from tools import *
import yaml
from scipy import stats

# Read data from config file
if os.path.exists("config_prediction.yaml"):
    with open('config_prediction.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        if data['model']['Unet_Sen1_Sen2']:

            sentinel1_folder = data['data_source']['Sentinel1']
            sentinel2_folder = data['data_source']['Sentinel2']
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder)) + glob("{}/*.tif".format(sentinel1_folder))
            sentinel_paths.sort() 
            model_path = data['model']['Unet_Sen1_Sen2']

        elif data['model']['Unet_Sen2']:

            sentinel2_folder = data['data_source']['Sentinel2']
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder))
            sentinel_paths.sort() 
            model_path = data['model']['Unet_Sen2']

        output_folder = data["output_folder"]

patching = True

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})
patch_size = model.input_shape[1]

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
        
        r_array = np.moveaxis(r_array, 0, -1)
        r_array = np.nan_to_num(r_array)
        
        if band_name not in ['VV', 'VH'] and idx != (len(sentinel_paths)-1):

            a,b = 0,1
            c,d = np.percentile(r_array, 1), np.percentile(r_array, 99)

            r_array = a+(r_array-c)*((b-a)/(d-c))
            r_array[r_array > 1] = 1
            r_array[r_array < 0] = 0
            
            # plt.hist(r_array.flatten(), bins = [0,200,400,1000,5000,10000])
            # plt.show()
            # plt.hist(r_array2.flatten(), bins = [0,0.2,0.4,0.6,0.8,1.0,2.0])
            # plt.show()  

        elif band_name in ['VV', 'VH'] and idx != (len(sentinel_paths)-1):

            a,b = -1,0
            c,d = np.percentile(r_array, 1), np.percentile(r_array, 99)

            r_array = a+(r_array-c)*((b-a)/(d-c))
            r_array[r_array < -1] = -1
            r_array[r_array > 0] = 0    

        bands_patches[band_name] = patchifyRasterAsArray(r_array, patch_size)

    patches_path = savePatchesPredict(bands_patches, output_folder)

patches_path = glob(r"{}/Crops/img/*.tif".format(output_folder))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

predictPatches(model, predict_datagen, sentinel_paths[4], output_folder)
