# Tensorflow application module for GAN (DCGAN) model

import tensorflow as tf
import numpy as np
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
from tensorflow.keras import layers
import time
from IPython import display


### CONSTANTS ###
seed = 123
num_examples_to_generate = 4
BATCH_SIZE = 256
EPOCHS = 5_000_000
INPUT_GENERATOR_SIZE = 32
lr_generator = 1e-4
lr_discriminator = 1e-4
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
OUTPUT_IMG_SIZE = (64, 64, 3)
generator_optimizer = tf.keras.optimizers.Adam(lr_generator)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_discriminator)
#################


def load_data(data_dir):
    """
    Get training and test data from data directory
    :param data_dir: directory of data
    :param batch_size: batch size
    :param output_img_size: output image size
    :return: training and test data
    """
    print("Loading data from directory: " + data_dir)
    print(len(os.listdir(data_dir)))
    abs_path = os.path.abspath(data_dir)
    print("data_dir: " + abs_path)
    # As the images are the true data, the labels are all 1s
    train_data = tf.keras.utils.image_dataset_from_directory(
        abs_path,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode='rgb',
        validation_split=0.1,
        subset="training",
        seed=seed,
        image_size=OUTPUT_IMG_SIZE[:2],
        batch_size=BATCH_SIZE)

    test_data = tf.keras.utils.image_dataset_from_directory(
        abs_path,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode='rgb',
        validation_split=0.1,
        subset="validation",
        seed=seed,
        image_size=OUTPUT_IMG_SIZE[:2],
        batch_size=BATCH_SIZE)

    # Standardize the data to be in the [0, 1] range by using a Rescaling layer
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    normalized_train_data = train_data.map(lambda x: (normalization_layer(x)))
    normalized_test_data = test_data.map(lambda x: (normalization_layer(x)))

    AUTOTUNE = tf.data.AUTOTUNE

    train_data = normalized_train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_data = normalized_test_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data, test_data


def make_generator_model():
    """
    Make generator model
    :return: generator model
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(INPUT_GENERATOR_SIZE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (8, 8), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (8, 8), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (8, 8), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (8, 8), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)


    model.summary()
    model.compile(optimizer=generator_optimizer,
                    loss=generator_loss,
                    metrics=['accuracy'])
    return model


def make_discriminator_model():
    """
    Make discriminator model
    :return: discriminator model
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=OUTPUT_IMG_SIZE))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))

    model.summary()
    model.compile(optimizer=discriminator_optimizer,
                  loss=cross_entropy,
                  metrics=['accuracy'])
    return model


def discriminator_loss(real_output, fake_output):
    """
    Calculate discriminator loss
    :param real_output: real output
    :param fake_output: fake output
    :return: discriminator loss
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    Calculate generator loss
    :param fake_output: fake output
    :return: generator loss
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def train(dataset, epochs):
    """
    Train model
    :param dataset: dataset
    :param epochs: epochs
    :return: None
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            # Train discriminator
            with tf.GradientTape() as disc_tape:
                noise = tf.random.normal([BATCH_SIZE, INPUT_GENERATOR_SIZE])
                generated_images = generator(noise, training=True)

                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Train generator
            with tf.GradientTape() as gen_tape:
                noise = tf.random.normal([BATCH_SIZE, INPUT_GENERATOR_SIZE])
                generated_images = generator(noise, training=True)

                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 np.random.normal(0, 1, (num_examples_to_generate, INPUT_GENERATOR_SIZE)))

        # Save the model every n epochs
        if (epoch + 1) % 1000 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} in {} sec'.format(epoch + 1, time.time() - start))
        print("-" * 50)


    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             np.random.normal(0, 1, (num_examples_to_generate, INPUT_GENERATOR_SIZE)))


def generate_and_save_images(model, epoch, test_input):
    """
    Generate and save images
    :param model: model
    :param epoch: epoch
    :param test_input: test input
    :return: None
    """
    predictions = model(test_input, training=False)

    plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.axis('off')

    plt.show()
    #for pred in predictions:
        # save image to png file
        #im = PIL.Image.fromarray(pred)
        #im.save("../result/generated_images/epoch_{}_{}.png".format(epoch, i))


def train_step(images):
    """
    Train step
    :param images: images
    :return: metrics (discriminator loss, generator loss)
    """
    noise = tf.random.normal([BATCH_SIZE, INPUT_GENERATOR_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Return losses
    return disc_loss, gen_loss


def display_image(epoch_no):
    """
    Display image
    :param epoch_no: epoch number
    :return: None
    """
    return PIL.Image.open(filename_image(epoch_no))


def filename_image(epoch_no):
    """
    Filename image
    :param epoch_no: epoch number
    :return: filename
    """
    return "../result/generated_image/image_{:04d}.png".format(epoch_no)

if __name__ == '__main__':
    data_dir = "../png_images/"
    generated_images_dir = "../result/generated_images/"
    model_dir = "../result/saved_models/"
    checkpoint_dir = './training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    train_dataset, test_dataset = load_data(data_dir)
    # if a model is already trained, load it
    if os.path.exists(model_dir):
        print("Training on existing model")
        generator = tf.keras.models.load_model(model_dir + "generator.h5")
        discriminator = tf.keras.models.load_model(model_dir + "discriminator.h5")
    else:
        generator = make_generator_model()
        discriminator = make_discriminator_model()

    checkpoint = tf.train.Checkpoint(   generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)

    if os.path.exists(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    print("Start training")
    train(train_dataset, EPOCHS)
    print("End training")

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    display_image(EPOCHS).show()

    # Save the model
    generator.save("../result/saved_models/generator")
    discriminator.save("../result/saved_models/discriminator")

    # Generate a GIF of all the saved images.
    with imageio.get_writer('../result/image.gif', mode='I') as writer:
        filenames = glob.glob(generated_images_dir + 'image_*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
