import os
from glob import glob
import yaml
from tools import *
from sklearn.model_selection import train_test_split
from unet import binary_unet
import rasterio
import random
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
from scipy import stats
import tensorflow as tf
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.layers import Input, Conv2D
from keras.models import Model

# Read data from config file
if os.path.exists("config_sen2.yaml"):
    with open('config_sen2.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['Solar']
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']

        output_folder = data["output_folder"]

preprocess_data = True
train_data = False

if preprocess_data:

    # get path of sentinel 1 and 2 tiles on codede server
    sen2_scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)

    [print(i) for i in sen2_scenes]

    # Get input data = Sentinel 2 + PV-Parks
    for tile in sen2_scenes:

        tile_name = '_'.join(tile.split("_")[-2:])

        sen2_path = glob(f"{tile}/*.tif") # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

        # Create mask as raster for each sentinel tile
        mask_path = rasterizeShapefile(sen2_path[2], solar_path, output_folder, tile_name, col_name="SolarPark")

        # all paths in one list
        sen2_mask = sen2_path + [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A MASK]

        bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}

        # Patchify all input data -> create smaller patches
        for idx, band in enumerate(sen2_mask):

            band_name = os.path.basename(band).split(".")[0]
            band_name = band_name.split("_")[-1]

            raster = rasterio.open(band)

            if raster.transform[0] != 10:
                raster = resampleRaster(band, 10)
                r_array = raster.ReadAsArray()
                r_array = np.expand_dims(r_array, axis=0)
            else:
                r_array = raster.read()[:,:10980,:10980]

            r_array = np.moveaxis(r_array, 0, -1)
            r_array = np.nan_to_num(r_array)

            print("Start patching with band: ", band_name)
            bands_patches[band_name] = patchifyRasterAsArray(r_array, 128)
        
        bands_count = len(bands_patches) - 1 # mask is not a band

        # Save patches in folder as raster file
        images_path, masks_path = savePatchesTrain(bands_patches, output_folder, tile_name)

        # Clear memory
        bands_patches = {}
        del r_array
        del raster

    # Data augmentation of saved patches
    seed_for_random = 42
    imageAugmentation(images_path, masks_path, bands_count)

if train_data:

    # Use patches as trainings data for model
    img_list = glob("{}/Crops/img/*.tif".format(output_folder))
    mask_list = glob("{}/Crops/mask/*.tif".format(output_folder))

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
    plt.imshow(X[0][:,:,:3].astype(np.uint8))

    for i in range(X.shape[0]):

        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(X[i][:,:,:3])
        plt.subplot(122)
        plt.imshow(y[i])
        plt.show()

    #Load model
    model = binary_unet(128,128,12)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_coef])

    model.fit(train_datagen, verbose=1, epochs=20, shuffle=True)

    #Save model for prediction
    model.save('pv_detection_Adam')

    print("Evaluate on test data")
    results = model.evaluate(test_datagen)
    print("test loss, test acc:", results)






