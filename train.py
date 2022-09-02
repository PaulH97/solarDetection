from sklearn.model_selection import train_test_split
from unet import binary_unet
import os
import rasterio
from sklearn.preprocessing import MinMaxScaler
import warnings
import rasterio
import random 
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

img_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\img"
mask_path = r"D:\Universität\Master_GeoInfo\Masterarbeit\data\Crops\All\mask"

img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)

img_list.sort()
mask_list.sort()

def updatePathIMG(file_name):
    return os.path.join(img_path, file_name)

def updatePathMask(file_name):
    return os.path.join(mask_path, file_name)

img_list_update = list(map(updatePathIMG, img_list))
mask_list_update = list(map(updatePathMask, mask_list))

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(img_list_update, mask_list_update, test_size = 0.10, random_state = 42)

train_datagen = CustomImageGenerator(X_train, y_train,(128,128))
test_datagen = CustomImageGenerator(X_test, y_test, (128,128))

batch_nr = random.randint(0, len(train_datagen))

X,y = train_datagen[batch_nr]

for i in range(X.shape[0]):

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(X[i][:,:,:3])
    plt.subplot(122)
    plt.imshow(y[i])
    plt.show()

smooth = 1.

from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

model = binary_unet(128,128,5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

results = model.fit(train_datagen, 
                    verbose=1, 
                    epochs=50, 
                    shuffle=True)

model.save('pv_detection')

