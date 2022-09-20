import rasterio

sen2_path = "path to sen2 directory"
output_folder = "path to output directory"

for band in sen2_path:

    raster = rasterio.open(band)
    raster_res = raster.transform[0]

    if raster_res != 10:  
        raster = resampleRaster(raster, output_folder, 10)
    