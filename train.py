from sklearn.model_selection import train_test_split
from unet import binary_unet
import os
import rasterio
import numpy as np
from sklearn.preprocessing import MinMaxScaler


img_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\img"
mask_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\mask"

img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)

img_list.sort()
mask_list.sort()

img_dataset = []
mask_dataset = []

scaler = MinMaxScaler()

for img in img_list:
    
    img_arr = rasterio.open(os.path.join(img_path, img)).read()
    img_arr = np.moveaxis(img_arr, 0, -1)
    img_arr = scaler.fit_transform(img_arr.reshape(-1, img_arr.shape[-1])).reshape(img_arr.shape)
    img_dataset.append(img_arr)

for mask in mask_list:
    
    mask_arr = rasterio.open(os.path.join(mask_path, mask)).read()
    mask_arr = np.moveaxis(mask_arr, 0, -1)
    mask_arr = scaler.fit_transform(mask_arr.reshape(-1, mask_arr.shape[-1])).reshape(mask_arr.shape)
    mask_dataset.append(mask_arr)

img_dataset = np.array(img_dataset)
mask_dataset = np.array(mask_dataset)

X_train, X_test, y_train, y_test = train_test_split(img_dataset, mask_dataset, test_size = 0.20, random_state = 42)

import random
import numpy as np
from matplotlib import pyplot as plt

img_nr = random.randint(0, len(X_train))
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(X_train[img_nr][:,:,:3])
plt.subplot(122)
plt.imshow(y_train[img_nr])
plt.show()

import pdb
pdb.set_trace()

model = binary_unet(128,128,5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

results = model.fit(X_train, y_train, 
                    batch_size=16, 
                    verbose=1, 
                    epochs=5, 
                    validation_data=(X_test,y_test), 
                    shuffle=False)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)


