

import tensorflow as tf
import numpy as np



# Load models

encoder = tf.keras.models.load_model("VAE_encoder.h5") 
decoder = tf.keras.models.load_model("VAE_decoder.h5")



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 

x_test = x_test.astype("float32") / 255.0  

x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))


encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)










