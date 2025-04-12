import tensorflow as tf
import numpy as np

def preprocess_image(image):
    # Converting to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((64, 64))
    
    # Converting to array and normalize
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    
    # Adding batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array