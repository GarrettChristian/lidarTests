"""
Following
https://keras.io/examples/generative/vae/
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import glob
from sklearn.model_selection import train_test_split


# sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



# Adapted from this tutorial:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
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

    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    # X = np.empty((self.batch_size, *self.dim))
    # y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        # X[i,] = np.load(ID)

        fromFile = np.fromfile(ID, dtype=np.ubyte)

        xyzArray = fromFile.reshape((int(np.shape(fromFile)[0]) // 3, 3))

        grid = np.zeros((256, 256), dtype=np.float32)

        for xyz in xyzArray:
            grid[xyz[0]][xyz[1]] = 1

        X[i,] = np.expand_dims(grid, axis=2)

    return X



def main():

    # PATH TO THE TRAINING FILES
    # path = "/media/garrett/Extreme SSD/rangeimgs/00/"
    # path = "/Volumes/Extreme SSD/rangeimgs/00/"
    # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/kitti/dataset/sequences/00/"
    # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/rangeimgs/00/"
    # path = "/p/lidarrealism/data/rangeimgs/"
    # path = "/p/lidarrealism/data/rangeimgs/00/"
    # path = "/p/lidarrealism/data/voxel4/00/"
    # path = "/p/lidarrealism/data/voxel4/"
    path = "/p/lidarrealism/data/voxels2/"
    # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/voxelTestScripts/voxels4/00/"
    # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/00/"
    # path = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTests/data/sets/voxels2/00/"
    # path = ""

    files = np.array(glob.glob(path + "*/*.bin", recursive = True))
    # files = np.array(glob.glob(path + "*.bin", recursive = True))
    print(np.shape(files))




    # Parameters
    # 'dim': (65536,),
    # 'dim': (64, 1024, 1)
    params = {'dim': (256, 256),
                'batch_size': 128,
                'n_channels': 1,
                'shuffle': True}


    # Datasets
    # labels = np.ones(np.shape(files)[0]) # Labels we don't actually use these 

    # train_data, test_data, train_labels, test_labels = train_test_split(
    #     files, labels, test_size=0.2, random_state=21
    # )

    # print(np.shape(train_data))

    # Generators
    training_generator = DataGenerator(files, files, **params)
    # validation_generator = DataGenerator(test_data, test_data, **params)

    # ---------------------------------------
    # MODEL

    latent_dim = 2
    
    # Encoder
    encoder_inputs = keras.Input(shape=(256, 256, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16384, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16384, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    print(decoder.summary())

    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    # print(vae.summary())

    history = vae.fit(training_generator, epochs=20, use_multiprocessing=True)
    # history = vae.fit(training_generator, validation_data=validation_generator, epochs=20)
    # history = vae.fit(training_generator, validation_data=validation_generator, epochs=20, use_multiprocessing=True)

    vae.save("pcdModel")


if __name__ == '__main__':
    main()
