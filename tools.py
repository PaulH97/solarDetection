import yaml
import rasterio
from patchify import patchify

vv = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\2021\CARD_BS_MC\33UVT\S1_CARD-BS-MC_202110_33UVT_VV.tif"
b2 = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\SENTINEL2X_20210915-000000-000_L3A_T33UVT_C_V1-2_FRC_B2.tif"
out = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\S1_CARD-BS-MC_202110_33UVT_VV_align.tif"


def resizeRaster(input_raster, output_path, raster_sample):
    """
    This function uses a sample raster to resize a raster file
    """
    with rio.open(input_raster) as src:

        name = os.path.basename(input_raster).split('.')[0] + "_resize.tif"    

        # store necessary information from sample raster
        raster_sample = rio.open(raster_sample)
        window = Window(0,0,raster_sample.width,raster_sample.height)

        # save content of raster as array for writing process
        array = src.read(1, window=window)
        
        # Update profile for new raster
        profile = src.profile
        profile['width'] = raster_sample.width
        profile['height'] = raster_sample.height
        profile['transform'] = src.window_transform(window)


        output = os.path.join(output_path, name)

        with rio.open(output, 'w', **profile) as dataset:
            dataset.write_band(1, array)
            print(f"Changed raster size from: {src.width, src.height} to {dataset.width, dataset.height}.")

# from osgeo import gdal, gdalconst

# inputfile = vv
# input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
# inputProj = input.GetProjection()
# inputTrans = input.GetGeoTransform()

# referencefile = b2
# reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
# referenceProj = reference.GetProjection()
# referenceTrans = reference.GetGeoTransform()
# bandreference = reference.GetRasterBand(1)    
# x = reference.RasterXSize 
# y = reference.RasterYSize

# outputfile = out
# driver= gdal.GetDriverByName('GTiff')
# output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
# output.SetGeoTransform(referenceTrans)
# output.SetProjection(referenceProj)

# gdal.ReprojectImage(input, output, inputProj, referenceProj, gdalconst.GRA_Bilinear)

# del output

# path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster"
# folders = os.listdir(path)
# for folder in folders:
#     files = os.listdir(os.path.join(path, folder))
#     for f in files:
#         name = os.path.join(path, folder, f)
#         print(name)
#         rename = os.path.join(path, folder, str(folder) + "_" + f)
#         print(rename)
#         os.rename(name, rename)


def load_img_as_array(path):
    # read img as array 
    import rasterio
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() 
    img_array = rasterio.open(path).read()
    img_array = np.moveaxis(img_array, 0, -1)
    img_array = scaler.fit_transform(img_array.reshape(-1, img_array.shape[-1])).reshape(img_array.shape)

    return img_array

