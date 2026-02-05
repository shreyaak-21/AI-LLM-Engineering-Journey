import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------
# Setup
# ----------------------------------
os.makedirs("outputs_cgan", exist_ok=True)

LATENT_DIM = 100
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 64

# ----------------------------------
# Load MNIST
# ----------------------------------
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(60000).batch(BATCH_SIZE)

# ----------------------------------
# Generator
# ----------------------------------
def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(NUM_CLASSES,))

    x = layers.Concatenate()([noise, label])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(28 * 28 * 1, activation="tanh")(x)
    img = layers.Reshape((28, 28, 1))(x)

    return tf.keras.Model([noise, label], img)

# ----------------------------------
# Discriminator
# ----------------------------------
def build_discriminator():
    img = layers.Input(shape=(28, 28, 1))
    label = layers.Input(shape=(NUM_CLASSES,))

    label_embedding = layers.Dense(28 * 28)(label)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    x = layers.Concatenate()([img, label_embedding])
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    validity = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model([img, label], validity)

generator = build_generator()
discriminator = build_discriminator()

# ----------------------------------
# Loss & Optimizers
# ----------------------------------
loss_fn = tf.keras.losses.BinaryCrossentropy()
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# ----------------------------------
# Training Step
# ----------------------------------
@tf.function
def train_step(images, labels):
    noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([fake_images, labels], training=True)

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        d_loss = (
            loss_fn(tf.ones_like(real_output), real_output)
            + loss_fn(tf.zeros_like(fake_output), fake_output)
        )

    grads_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    grads_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

# ----------------------------------
# Save generated digits
# ----------------------------------
def save_images(epoch):
    noise = tf.random.normal((10, LATENT_DIM))
    labels = tf.one_hot(np.arange(10), NUM_CLASSES)

    generated = generator([noise, labels], training=False)
    generated = (generated + 1) / 2.0

    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(generated[i, :, :, 0], cmap="gray")
        plt.title(str(i))
        plt.axis("off")

    plt.savefig(f"outputs_cgan/epoch_{epoch}.png")
    plt.close()

# ----------------------------------
# Training Loop
# ----------------------------------
for epoch in range(EPOCHS):
    for img_batch, label_batch in dataset:
        if img_batch.shape[0] != BATCH_SIZE:
            continue
        train_step(img_batch, label_batch)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    if (epoch + 1) % 5 == 0:
        save_images(epoch + 1)