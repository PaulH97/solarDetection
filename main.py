import os
from glob import glob
import yaml
from tools import *
from sklearn.model_selection import train_test_split
from unet import binary_unet
from sklearn.preprocessing import MinMaxScaler
import rasterio
import random 
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator

# Read data from config file
if os.path.exists("config_training.yaml"):
    with open('config_training.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['AOI']
        sentinel1_name = data['satellite']['Sentinel1']['name']
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel1_pl = data['satellite']['Sentinel1']['processing_level'] 
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel1_pt = data['satellite']['Sentinel1']['product_type']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']

        output_folder = data["output_folder"]

find_data = True
preprocess_data = False
train_data = False

if find_data:

    # get path of sentinel 1 and 2 tiles on codede server
    sen2_scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)
    sen1_scenes = ScenceFinderAOI(aoi_path, sentinel1_name, sentinel1_pl, sentinel1_pt, start_date, end_date, cloud_cover)

    sceneList = []
    for sen2 in sen2_scenes:
        for sen1 in sen1_scenes:
            sen1_id = str('_'.join(sen1.split("_")[-2:]))
            sen2_id = str('_'.join(sen2.split("_")[-2:]))

            if sen1_id == sen2_id:
                sceneList.append([sen1,sen2])

    final_list = filterScenes(sceneList)

    [print(i) for i in final_list]
    
if preprocess_data:
    
    # Get input data = Sentinel 1 + Sentinel 2 + PV-Parks
    for tile in final_list:

        sen1_path = glob(f"{tile[0]}/*.tif") # VH VV
        sen2_path = glob(f"{tile[1]}/*.tif") # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

        # Create mask as raster for each sentinel tile
        mask_path = rasterizeShapefile(sen2_path[2], solar_path, output_folder, col_name="SolarPark")

        import pdb 
        pdb.set_trace()

        # all paths in one list
        sen2_sen1_mask = sen2_path + sen1_path +  [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A VH VV MASK] 

        bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}

        # Patchify all input data -> create smaller patches
        for idx, band in enumerate(sen2_sen1_mask):

            band_name = os.path.basename(band).split(".")[0]
            if idx != 12:
                band_name = band_name.split("_")[-1]

            print("Start patching with band: ", band_name)
            raster = rasterio.open(band)
            
            if raster.transform[0] != 10:  
                raster = resampleRaster(band, "output_folder", 10)
                r_array = raster.ReadAsArray()
                r_array = np.expand_dims(r_array, axis=0)
            else:
                r_array = raster.read()[:,:10980,:10980]
            
            r_array = np.moveaxis(r_array, 0, -1)

            bands_patches[band_name] = patchifyRasterAsArray(r_array, 128)

        # Save patches in folder as raster file 
        images_path, masks_path = savePatches(bands_patches, output_folder, sen2_sen1_mask[0])

        # Data augmentation of saved patches
        seed_for_random = 42
        imageAugmentation(images_path, masks_path)

if train_data:
    # Use patches as trainings data for model 
    img_list = glob(r"{}\Crops\img\*.tif".format(output_folder))
    mask_list = glob(r"{}\Crops\mask\*.tif".format(output_folder))

    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.10, random_state = 42)

    # Load images and masks with an custom data generator - for performance reason
    train_datagen = CustomImageGenerator(X_train, y_train,(128,128))
    test_datagen = CustomImageGenerator(X_test, y_test, (128,128))

    # sanity check
    batch_nr = random.randint(0, len(train_datagen))
    X,y = train_datagen[batch_nr]

    for i in range(X.shape[0]):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(X[i][:,:,:3])
        plt.subplot(122)
        plt.imshow(y[i])
        plt.show()

    # Load model 
    model = binary_unet(128,128,12)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()
    
    # results = model.fit(train_datagen, 
    #                     verbose=1, 
    #                     epochs=50, 
    #                     shuffle=True)
    
    # Save model for prediction 
    # model.save('pv_detection')

import os
from glob import glob
import yaml
from tools import *
from sklearn.model_selection import train_test_split
from unet import binary_unet
from sklearn.preprocessing import MinMaxScaler
import rasterio
import random 
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
from keras import backend as K


# Read data from config file
if os.path.exists("config_training.yaml"):
    with open('config_training.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['Solar']
        sentinel1_name = data['satellite']['Sentinel1']['name']
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel1_pl = data['satellite']['Sentinel1']['processing_level'] 
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel1_pt = data['satellite']['Sentinel1']['product_type']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']

        output_folder = data["output_folder"]

find_data = False
preprocess_data = False
train_data = True

if find_data:

    # get path of sentinel 1 and 2 tiles on codede server
    sen2_scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)
    sen1_scenes = ScenceFinderAOI(aoi_path, sentinel1_name, sentinel1_pl, sentinel1_pt, start_date, end_date, cloud_cover)

    sceneList = []
    for sen2 in sen2_scenes:
        for sen1 in sen1_scenes:
            sen1_id = str('_'.join(sen1.split("_")[-2:]))
            sen2_id = str('_'.join(sen2.split("_")[-2:]))

            if sen1_id == sen2_id:
                sceneList.append([sen1,sen2])

    final_list = filterScenes(sceneList)

    [print(i) for i in final_list]
    
if preprocess_data:
    
    # Get input data = Sentinel 1 + Sentinel 2 + PV-Parks
    for tile in final_list:
        
        tile_name = '_'.join(tile[0].split("_")[-2:])

        sen1_path = glob(f"{tile[0]}/*.tif") # VH VV
        sen2_path = glob(f"{tile[1]}/*.tif") # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

        # Create mask as raster for each sentinel tile
        mask_path = rasterizeShapefile(sen2_path[2], solar_path, output_folder, tile_name, col_name="SolarPark")

        # all paths in one list
        sen2_sen1_mask = sen2_path + sen1_path +  [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A VH VV MASK] 

        bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
        
        # Patchify all input data -> create smaller patches
        for idx, band in enumerate(sen2_sen1_mask):

            band_name = os.path.basename(band).split(".")[0]
            if idx != 12:
                band_name = band_name.split("_")[-1]

            print("Start patching with band: ", band_name)
            raster = rasterio.open(band)
            
            if raster.transform[0] != 10:  
                raster = resampleRaster(band, "output_folder", 10)
                r_array = raster.ReadAsArray()
                r_array = np.expand_dims(r_array, axis=0)
            else:
                r_array = raster.read()[:,:10980,:10980]
            
            r_array = np.moveaxis(r_array, 0, -1)
            
            bands_patches[band_name] = patchifyRasterAsArray(r_array, 128)
            
        # Save patches in folder as raster file 
        images_path, masks_path = savePatches(bands_patches, output_folder, tile_name)
        
        # Clear memory
        bands_patches = {}
        del r_array 
        del raster
        
    # Data augmentation of saved patches
    seed_for_random = 42
    imageAugmentation(images_path, masks_path)

if train_data:

    # Use patches as trainings data for model 
    img_list = glob(r"{}/Crops/img/*.tif".format(output_folder))
    mask_list = glob(r"{}/Crops/mask/*.tif".format(output_folder))
    
    img_list.sort()
    mask_list.sort()
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, random_state = 42)

    # Load images and masks with an custom data generator - for performance reason
    train_datagen = CustomImageGenerator(X_train, y_train,(128,128))
    test_datagen = CustomImageGenerator(X_test, y_test, (128,128))

    # sanity check
    batch_nr = random.randint(0, len(train_datagen))
    X,y = train_datagen[batch_nr]

    for i in range(X.shape[0]):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(X[i][:,:,:3])
        plt.subplot(122)
        plt.imshow(y[i])
        plt.show()
            
    # Load model 
    model = binary_unet(128,128,12)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()
    
    results = model.fit(train_datagen, verbose=1, epochs=25, shuffle=True)
    
    # Save model for prediction 
    model.save('pv_detection')

