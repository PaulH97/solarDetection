
import rasterio 
import numpy as np
from glob import glob
from patchify import patchify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# load sentinel 1 & 2 band paths as list
sen1_path_resize = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel1\S1_CARD-BS-MC_202110_33UVT_*_resize.tif")
sen2_path = glob(r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Sentinel2\L3_WASP\2021\33UVT\SENTINEL2X_20210915-000000-000_L3A_T33UVT_C_V1-2_FRC_B*.tif")
sen2_path.sort(reverse=True) # bands 4 3 2

# mask path as list
mask = [r"D:\Universität\Master_GeoInfo\Masterarbeit\data\SolarParks\raster\33UVT\pv_anlagen.tif"]

# alle paths in one list
sen2_sen1_mask = sen2_path + sen1_path_resize + mask # [red, green, blue, vv, vh, mask]

# define patch size 
patch_size = 128
scaler = MinMaxScaler()
img_dataset = []

for band in sen2_sen1_mask:

    raster = rasterio.open(band)
    array = raster.read().reshape(raster.width,raster.height,1) # (10980,10980,1)

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

# visualize 
import random
import numpy as np
from matplotlib import pyplot as plt

idx_list = []

for idx, label in enumerate(labels):
    if  np.count_nonzero(label == 1):
        idx_list.append(idx)

image_number = random.randint(0, len(img_dataset_stack))
image_number = random.choice(idx_list)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_dataset_stack[image_number][:,:,0])
plt.subplot(122)
plt.imshow(label_dataset[image_number])
plt.show()

img_dataset_stack = np.array(img_dataset_stack)
label_dataset = np.array(label_dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_dataset_stack, label_dataset, test_size = 0.20, random_state = 42)

from unet import binary_unet

model = binary_unet(128,128,5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

results = model.fit(X_train, y_train, 
                    batch_size=16, 
                    verbose=1,
                    monitor='val_loss', 
                    epochs=5, 
                    validation_data=(X_test,y_test), 
                    shuffle=False)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)


