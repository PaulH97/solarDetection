from glob import glob
import rasterio as rio 
from rasterio.windows import Window
import os

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


input_raster = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT\pv_anlagen.tif"

raster_sample = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\33UVT.tif"

output_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT"

resizeRaster(input_raster, output_path, raster_sample)