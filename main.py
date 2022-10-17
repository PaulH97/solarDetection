import os
from glob import glob
import yaml
from tools import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from unet import binary_unet, binary_unet2
import rasterio
import random
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
from scipy import stats
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        
        model_sen12 = data['model']['Sentinel1+2']
        patch_size = data['model']['patch_size']
        optimizer = data['model']['optimizer']
        loss_function = data['model']['loss_function']
        epochs = data['model']['epochs']

        output_folder = data["output_folder"]

preprocess_data = True
train_data = True

if preprocess_data:
    if model_sen12:
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

        tiles_list = filterSen12(sceneList, filterDate=False)
        
        print("Found the following Sentinel 1 and Sentinel 2 scenes")
        [print(i) for i in tiles_list]
    
    else:
        sceneList = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)
        tiles_list = filterSen2(sceneList, filterDate=False)
        
        print("Found the following Sentinel 2 scenes")
        [print(i) for i in tiles_list]

    # Get input data = Sentinel 1 + Sentinel 2 + PV-Parks
    for tile in tiles_list:
        # get tile name and path for each band
        if model_sen12:
            tile_name = '_'.join(tile[0].split("_")[-2:])
            sen_path = glob(f"{tile[1]}/*.tif") + glob(f"{tile[0]}/*.tif") 
            sen_path.sort() # VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 
        else:
            tile_name = '_'.join(tile.split("_")[-2:])
            sen_path = glob(f"{tile}/*.tif") 
            sen_path.sort() # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A
        
        print("Start with tile: ", tile_name)

        # Create mask as raster for each sentinel tile
        mask_path = rasterizeShapefile(sen_path[2], solar_path, output_folder, tile_name, col_name="SolarPark")

        # all paths in one list
        sen_mask = sen_path + [mask_path] # [(VH VV) B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A  MASK]

        bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
        scaler = MaxAbsScaler()
        
        # Patchify all input data -> create smaller patches
        for idx, band in enumerate(sen_mask):

            band_name = os.path.basename(band).split(".")[0].split("_")[-1]
            print("Start with band: ", band_name)
            
            raster = rasterio.open(band)

            if raster.transform[0] != 10:
                raster = resampleRaster(band, 10)
                r_array = raster.ReadAsArray()
                r_array = np.expand_dims(r_array, axis=0)
            else:
                r_array = raster.read()[:,:10980,:10980]

            r_array = np.moveaxis(r_array, 0, -1)
            r_array = np.nan_to_num(r_array)

            if band_name not in ['VV', 'VH'] and idx != (len(sen_mask)-1):

                a,b = 0,1
                c,d = np.percentile(r_array, 1), np.percentile(r_array, 99)

                r_array = a+(r_array-c)*((b-a)/(d-c))
                r_array[r_array > 1] = 1
                r_array[r_array < 0] = 0
                
                # plt.hist(r_array.flatten(), bins = [0,200,400,1000,5000,10000])
                # plt.show()
                # plt.hist(r_array2.flatten(), bins = [0,0.2,0.4,0.6,0.8,1.0,2.0])
                # plt.show()  

            elif band_name in ['VV', 'VH'] and idx != (len(sen_mask)-1):
    
                a,b = -1,0
                c,d = np.percentile(r_array, 1), np.percentile(r_array, 99)

                r_array = a+(r_array-c)*((b-a)/(d-c))
                r_array[r_array < -1] = -1
                r_array[r_array > 0] = 0

                # plt.hist(r_array.flatten(), bins = [-40, -30, -20, -10, 0, 10])
                # plt.show()
                # plt.hist(r_array2.flatten(), bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2])
                # plt.show()           

            bands_patches[band_name] = patchifyRasterAsArray(r_array, patch_size)

        # Save patches in folder as raster file
        images_path, masks_path = savePatchesTrain(bands_patches, output_folder)

        # Clear memory
        bands_patches = {}
        del r_array
        del raster

    # Data augmentation of saved patches
    imageAugmentation(images_path, masks_path)
    print("---------------------")

if train_data:

    # Use patches as trainings data for model
    img_list = glob("{}/Crops/img/*.tif".format(output_folder))
    mask_list = glob("{}/Crops/mask/*.tif".format(output_folder))

    img_list.sort()
    mask_list.sort()

    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, random_state = 42)
    
    # Load images and masks with an custom data generator - for performance reason
    patch_array = load_img_as_array(X_train[0])
    patch_xy = (patch_array.shape[0], patch_array.shape[1])
    b_count = patch_array.shape[-1]
    
    train_datagen = CustomImageGenerator(X_train, y_train, patch_xy, b_count)
    test_datagen = CustomImageGenerator(X_test, y_test, patch_xy, b_count)

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

    # TODO test model with diffrent parameters
    # 256x256 Adam 0.001 jac_dist dice_coeff
    # 256x256 SGD 0.001 jac_dist dice_coeff
    # 128x128 Adam 0.001 jac_dist dice_coeff
    # 128x128 SGD 0.001 jac_dist dice_coeff            
    # 
                                                                                                                                                                                                                                                                                                                                                            
    #Load model
    model = binary_unet2(patch_xy[0], patch_xy[1], b_count)  
    # optimizer = getattr(importlib.import_module("tensorflow.keras.optimizers"), optimizer)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # or SGD
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[dice_coef])
    # Model fit 
    model.fit(train_datagen, verbose=1, epochs=epochs)

    # Save model for prediction
    # "Sentinel2_optimizer_loss_epochs"
    model_name = optimizer + "_" + loss_function + "_" + str(epochs) 
    model.save(model_name)

    #model = tf.keras.models.load_model('pv_detection_Adam_cutoff', compile=False, custom_objects={'dice_coef': jaccard_distance_loss})

    results = model.predict(test_datagen) # f.eg.(288,128,128,1)
    results = (results > 0.5).astype(np.uint8) 

    for i in range((test_datagen[0][1].shape[0])):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(test_datagen[0][1][i])
        plt.subplot(122)
        plt.imshow(results[i])
        plt.show()







