import numpy as np
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential

from PIL import Image
from PIL import ImageOps


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

import glob, os
import struct

# ----------------------------------------------------------------------------------------------------

# https://blog.keras.io/building-autoencoders-in-keras.html
def create_model():
  model = Sequential()

  # Encoder
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) # orig 16
  model.add(layers.MaxPooling2D((2, 2), padding='same'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.MaxPooling2D((2, 2), padding='same'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.MaxPooling2D((2, 2), padding='same'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.MaxPooling2D((2, 2), padding='same'))
  
  # Decoder 
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # orig 8
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) # orig 16
  model.add(layers.UpSampling2D((2, 2)))
  model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

  return model

  


# ----------------------------------------------------------------------------------------------------

# Adapted from this tutorial:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://gist.github.com/twolodzko/aa4f4ad52f16c293df40342929b025a4?short_path=4d6ba34
# one above is really cool example of how the noise version works
class DataGenerator(keras.utils.Sequence):
  def __init__(self, list_IDs, labels, batch_size=100, dim=(2070272), n_channels=1, shuffle=True):
      'Initialization'
      self.dim = dim
      self.batch_size = batch_size
      self.labels = labels
      self.list_IDs = list_IDs
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.on_epoch_end()

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    # np.empty((self.batch_size, *self.dim, self.n_channels))
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size, *self.dim))
    # y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        # X[i,] = np.load(ID)
        
        image = Image.open(ID)
        imageGrey = ImageOps.grayscale(image)
        imageGreyArray = np.array(imageGrey)
        imageGreyArrayNorm = imageGreyArray.astype('float32') / 255
        X[i,] = np.expand_dims(imageGreyArrayNorm, axis=2)
        # print(np.shape(X[i,]))
        
        # Store class
        # y[i] = self.labels[ID]
        # y[i] = 1

    # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    # return X, y
    return X, X

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y


# ----------------------------------------------------------------------------------------------------

def main():

  # autoencoder = AnomalyDetector()
  autoencoder = create_model()
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

  # PATH TO THE TRAINING FILES
  # path = "/media/garrett/Extreme SSD/semKitti/dataset/sequences/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
  # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/"
  path = "/p/lidarrealism/data/rangeimgs/00/"

  files = np.array(glob.glob(path + "*.png", recursive = True))
  print(np.shape(files))

  # Parameters
  # 'dim': (65536,),
  # 'dim': (64, 1024, 1)
  params = {'dim': (64, 1024, 1),
            'batch_size': 100,
            'n_channels': 1,
            'shuffle': True}


  # Datasets
  labels = np.ones(np.shape(files)[0]) # Labels we don't actually use these 

  train_data, test_data, train_labels, test_labels = train_test_split(
      files, labels, test_size=0.2, random_state=21
  )

  print(np.shape(train_data))

  # Generators
  training_generator = DataGenerator(train_data, train_data, **params)
  validation_generator = DataGenerator(test_data, test_data, **params)

  history = autoencoder.fit(training_generator, validation_data=validation_generator, epochs=20, use_multiprocessing=True)

  autoencoder.save("pcdModel")


if __name__ == '__main__':
    main()