from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.models import Model

#Import vgg model by not defining an input shape. 
vgg_model = VGG16(include_top=False, weights='imagenet')
print(vgg_model.summary())

#Get the dictionary of config for vgg16
vgg_config = vgg_model.get_config()

# Change the input shape to new desired shape
height = 256
width = 256
bands = 5
# Update inpiut shape in model config
vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)