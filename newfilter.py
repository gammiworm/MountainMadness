import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers

def newFilter():
    filter = [
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2))]
    
    return filter