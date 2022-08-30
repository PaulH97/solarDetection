import numpy as np
import random
import os
from tools import load_img_as_array
import rasterio
import tifffile as tiff
from matplotlib import pyplot as plt

images_path="data/images/" #path to original images
masks_path = "data/masks/"
img_augmented_path="data/augmented/augmented_images/" # path to store aumented images
msk_augmented_path="data/augmented/augmented_masks/" # path to store aumented images

seed_for_random = 42

def rotation90(image, seed):
    random.seed(seed)
    r_image = np.rot90(image)
    return r_image

def h_flip(image, seed):
    hflipped_img= np.fliplr(image)
    return hflipped_img

def v_flip(image, seed):
    vflipped_img= np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img

transformations = {'rotate': rotation90, 'horizontal flip': h_flip,'vertical flip': v_flip, 'vertical shift': v_transl, 'horizontal shift': h_transl}         

images=[] 
masks=[]

for im in os.listdir(images_path):      
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  
    masks.append(os.path.join(masks_path,msk))

for i in range(len(images)): 
    
    image = images[i]
    mask = masks[i]

    original_image = load_img_as_array(image)
    original_mask = load_img_as_array(mask)
    
    key = random.choice(list(transformations)) #randomly choosing method to call
    seed = random.randint(1,100)  #Generate seed to supply transformation functions. 
    transformed_image = transformations[key](original_image, seed)
    transformed_mask = transformations[key](original_mask, seed)

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(original_image[:,:,:3])
    plt.subplot(122)
    plt.imshow(transformed_image[:,:,:3])
    plt.show()

    import pdb
    pdb.set_trace()

    new_image_path= "%s/augmented_image_%s.tif" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.tif" %(msk_augmented_path, i)   #Do not save as JPG
    
    new_img = rasterio.open(new_image_path,'w', driver='Gtiff',
                width=transformed_image.shape[0], height=transformed_image.shape[1],
                count=5,
                dtype=rasterio.float32)
    
    for band in transformed_image.shape[-1]:
        new_img.write(transformed_image[:,:,band])
    new_img.close() 
    
    tiff.imwrite(new_mask_path, transformed_mask)

    