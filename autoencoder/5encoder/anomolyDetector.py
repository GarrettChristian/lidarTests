import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras


class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(524288, activation="relu"),
      layers.Dense(262144, activation="relu"),
      layers.Dense(131072, activation="relu"),
      # layers.Dense(65536, activation="relu"),
      # layers.Dense(32768, activation="relu"),
      # layers.Dense(16384, activation="relu"),
      # layers.Dense(8192, activation="relu"),
      # layers.Dense(1024, activation="relu"),
      # layers.Dense(512, activation="relu"),
      # layers.Dense(256, activation="relu")
      ])

    self.decoder = tf.keras.Sequential([
      # layers.Dense(256, activation="relu"),
      # layers.Dense(512, activation="relu"),
      # layers.Dense(1024, activation="relu"),
      # layers.Dense(8192, activation="relu"),
      # layers.Dense(16384, activation="relu"),
      # layers.Dense(32768, activation="relu"),
      # layers.Dense(65536, activation="relu"),
      layers.Dense(131072, activation="relu"),
      layers.Dense(262144, activation="relu"),
      layers.Dense(524288, activation="relu"),
      layers.Dense(1035136, activation="sigmoid")
      ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

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
    # X = np.empty((self.batch_size, *self.dim, self.n_channels))
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size, *self.dim))
    # y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        # X[i,] = np.load(ID)
        pcdi = np.fromfile(ID, dtype=np.float32)
        pcdi = pcdi.reshape((int(np.shape(pcdi)[0]) // 4, 4))
        pcdi = np.delete(pcdi, np.s_[2::], 1)
        pcdi = pcdi.reshape(np.size(pcdi),)
        X[i,] = np.pad(pcdi, (0,1035136 - int(np.shape(pcdi)[0])), 'constant', constant_values=(0))
        
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