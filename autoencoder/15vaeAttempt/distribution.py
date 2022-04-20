

import tensorflow as tf
from tensorflow import keras    
import numpy as np

# vae = load_model('vaeExample.h5', custom_objects={'latent_dim': latent_dim, 'epsilon_std': epsilon_std, 'vae_loss': vae_loss})
# vae = load_weights('vaeExample.h5')
# vae.summary()
encoder = tf.keras.models.load_model("encoderVaeModel") 
decoder = tf.keras.models.load_model("decoderVaeModel")
# vae = VAE(encoder, decoder)

import matplotlib.pyplot as plt


def plot_latent_space(decoder, n=5, figsize=15):
    # display a n*n 2D manifold of digits
    scale = 1.0
    figure = np.zeros((64 * n, 1024 * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(64, 1024)
            figure[
                i * 64 : (i + 1) * 64,
                j * 1024 : (j + 1) * 1024,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = 64 // 2
    end_range = n * 1024 + start_range
    pixel_range = np.arange(start_range, end_range, 1024)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(decoder)

