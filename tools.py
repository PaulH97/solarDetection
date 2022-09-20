import yaml
import rasterio
from patchify import patchify
from rasterio import features
from rasterio.enums import MergeAlg
import numpy as np
import geopandas as gdp
import gdal
import os

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

def resampleRaster(raster_path, output_folder, resolution):
    
    name = os.path.basename(raster_path)
    output_file = os.path.join(output_folder, name + "_resample.tif")

    raster = gdal.Open(raster_path)
    ds = gdal.Warp(output_file, raster, xRes=resolution, yRes=resolution, resampleAlg="bilinear", format="GTiff")

    return

def rasterizeShapefile(raster_path, vector_path, output_path):
    
    raster = rasterio.open(raster_path)
    vector = gdp.read_file(vector_path)
    
    geom_value = ((geom,value) for geom, value in zip(vector.geometry, vector['gridcode']))
    osm_res = 10
    crs    = rasterio.crs.CRS.from_epsg(vector.crs.to_epsg())
    transform = raster.transform

    r_out = os.path.join(output_path, os.path.basename(vector_path).split(".")[0] +".tif")
    with rasterio.open(r_out, 'w+', driver='GTiff',
            height = raster.height, width = raster.width,
            count = 1, dtype="int16", crs = crs, transform=transform) as rst:
        out_arr = rst.read(1)
        rasterized = features.rasterize(shapes=geom_value, fill=0, out=out_arr, transform = rst.transform)
        rst.write_band(1, rasterized)
    rst.close()