from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.models import Model


def binary_unet_model(img_width, img_height, img_bands):

    #Import vgg model by not defining an input shape. 
    vgg_model = VGG16(include_top=False, weights='imagenet')

    #Get the dictionary of config for vgg16
    vgg_config = vgg_model.get_config()

    # Change the input shape to new desired shape
    height = img_height
    width = img_width
    bands = img_bands
    # Update inpiut shape in model config
    vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, height, width, bands)

    #Create new model with the updated configuration
    vgg_updated = Model.from_config(vgg_config)
    vgg_updated_ln = [layer.name for layer in vgg_updated.layers]

    # only need to update this first conv layer in the model - rest of the layers will have the same weights as the original model
    first_conv_name = vgg_updated_ln[1]

    def copy_weights(weights, method, multiplicator):

        if method == "avg":
            avg_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)
            weights_copied = np.tile(avg_weights, (multiplicator, 1))
        elif method == "max":
            max_weights = np.max(weights, axis=-2).reshape(weights[:,:,-1:,:].shape) 
            weights_copied = np.tile(max_weights, (multiplicator, 1))
        else:
            weights_copied = np.tile(weights, (multiplicator, 1))
    
        return(weights_copied)


    for layer in vgg_model.layers:
        if layer.name in vgg_updated_ln:
        
            if layer.get_weights() != []: 
                target_layer = vgg_updated.get_layer(layer.name)
            
                if layer.name in first_conv_name:   
                    weights = layer.get_weights()[0]
                    biases  = layer.get_weights()[1]

                    weights_extra_channels = np.concatenate((weights,   
                                                        copy_weights(weights, "avg", 2)), 
                                                        axis=-2)
                                                        
                    target_layer.set_weights([weights_extra_channels, biases])  
                    target_layer.trainable = True  
            
                else:
                    target_layer.set_weights(layer.get_weights())   
                    target_layer.trainable = True  

    return vgg_updated
