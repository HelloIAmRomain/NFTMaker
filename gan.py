"""
This program is a GAN that generates images 400x400x3 like NFTs apes.
"""

import os
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import time
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU, InputLayer, GlobalAveragePooling2D, Reshape, Conv2DTranspose, ReLU
import argparse
from tqdm import tqdm

@tf.function
def load_image_png(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    return image


def load_dataset(image_dir):
    image_paths = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Augment the dataset with horizontal flips (x2 the size)
    dataset = dataset.concatenate(dataset.map(lambda x: tf.image.flip_left_right(x)))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Remove the last batch if it's smaller than the batch size
    dataset = dataset.filter(lambda x: tf.shape(x)[0] == batch_size)
    return dataset
        


############################################################################################################
# Variables
############################################################################################################
generator_in_channels = 128
discriminator_in_channels = (400, 400, 3)
batch_size = 8
epochs = 1000
n_train_generator = 64


########################################################################################################################
# Model creation
########################################################################################################################


discriminator = tf.keras.Sequential([
    InputLayer(input_shape=(400, 400, 3)),
    Conv2D(128, 5, strides=2, padding='same'),
    ReLU(),
    Conv2D(128, 5, strides=2, padding='same'),
    Conv2D(64, (3,3 ), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

print(discriminator.summary())


# Generator model (output shape: 400x400x3)
generator = tf.keras.Sequential([
    InputLayer(input_shape=(generator_in_channels,)),
    Dense(10_000, activation='relu'),
    Reshape((25, 25, 16)),
    Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(3, (3, 3), strides=(4, 4), padding='same', activation='sigmoid'),
])

print(generator.summary())


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name='discriminator_loss')
        self.g_loss_tracker = tf.keras.metrics.Mean(name='generator_loss')
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (via the gan model, where the discriminator weights are frozen)
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor losses
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(), "g_loss": self.g_loss_tracker.result()}
    
    def train_generator(self):
        """
        Train the generator only
        """
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (via the gan model, where the discriminator weights are frozen)
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss




########################################################################################################################
# Training
########################################################################################################################
# if a model already exists and --continue-training is True, load it
parser = argparse.ArgumentParser()
parser.add_argument('--continue-training', '-c', default=True, type=bool)
args = parser.parse_args()

if args.continue_training and os.path.exists('generator_weights.h5'):
    generator.load_weights('generator_weights.h5')

if args.continue_training and os.path.exists('discriminator_weights.h5'):
    discriminator.load_weights('discriminator_weights.h5')

train_dataset = load_dataset(image_dir='png_images/')

gan = GAN(discriminator=discriminator,
          generator=generator, 
          latent_dim=generator_in_channels)


gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss_fn=tf.keras.losses.BinaryCrossentropy()
            )


for epoch in tqdm(range(epochs)):
    print('Epoch: ', epoch)
    for step, real_images in enumerate(train_dataset):
        gan.train_step(real_images)
    
    #for tr_gen in range(n_train_generator):
    #    gan.train_generator()

    print('Discriminator loss: ', gan.d_loss_tracker.result())
    print('Generator loss: ', gan.g_loss_tracker.result())
    print('Saving weights...')
    generator.save_weights('generator_weights.h5')
    discriminator.save_weights('discriminator_weights.h5')

    # Save a generated image every epoch
    random_latent_vectors = tf.random.normal(shape=(1, generator_in_channels))
    generated_image = generator(random_latent_vectors)
    generated_image = generated_image.numpy()
    generated_image = generated_image.reshape(400, 400, 3)
    generated_image = generated_image * 255
    generated_image = generated_image.astype(np.uint8)
    im = Image.fromarray(generated_image)
    im.save('generated_images/generated_image_' + str(epoch) + '.png')

    # Save the model every epoch
    #generator.save('generator_model.h5')
    #discriminator.save('discriminator_model.h5')

    # Reset metrics every epoch
    gan.d_loss_tracker.reset_states()
    gan.g_loss_tracker.reset_states()

    print('Epoch: ', epoch, ' completed')




########################################################################################################################
# Results
########################################################################################################################

# Plot the losses
plt.plot(gan.history.history["d_loss"], label="discriminator")
plt.plot(gan.history.history["g_loss"], label="generator")
plt.legend()
plt.show()

# Plot some generated images
random_latent_vectors = tf.random.normal(shape=(10, generator_in_channels))
generated_images = gan.generator(random_latent_vectors)
generated_images = generated_images * 255
generated_images = generated_images.numpy()

fig, axs = plt.subplots(2, 5, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(generated_images[i])
    ax.axis("off")
plt.show()
