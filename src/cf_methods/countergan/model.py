import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    UpSampling2D,
    LeakyReLU,
)
from tensorflow.keras.layers import (
    Conv2DTranspose,
    Reshape,
    GaussianNoise,
    ActivityRegularization,
    Add,
)


def create_convolutional_autoencoder(in_shape=(224, 224, 1)):
    input_img = Input(shape=in_shape)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 224 -> 112

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)  # 112 -> 56

    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation="linear", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    optimizer = optimizers.Adam(learning_rate=0.001)

    autoencoder.compile(optimizer, "mse")

    return autoencoder


def create_generator(in_shape=(224, 224, 1), residuals=True):
    """Define and compile the residual generator of the CounteRGAN."""

    generator_input = Input(shape=in_shape, name="generator_input")
    generator = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(generator_input)
    generator = LeakyReLU(negative_slope=0.2)(generator)
    generator = Dropout(0.2)(generator)
    # print(generator.shape)

    generator = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(generator)
    generator = LeakyReLU(negative_slope=0.2)(generator)
    generator = Dropout(0.2)(generator)
    temp_shape = generator.shape[1:]
    # print(generator.shape)

    generator = Flatten()(generator)

    # Deconvolution
    generator = Dense(np.prod(temp_shape).item())(generator)
    generator = LeakyReLU(negative_slope=0.2)(generator)
    generator = Reshape(temp_shape)(generator)
    # print(generator.shape)

    # upsample to 14x14
    generator = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="valid")(generator)
    generator = LeakyReLU(negative_slope=0.2)(generator)
    # print(generator.shape)

    # upsample to 28x28
    generator = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="valid")(generator)
    generator = LeakyReLU(negative_slope=0.2)(generator)
    # print(generator.shape)

    generator = Conv2D(1, (4, 4), activation="tanh", padding="same")(generator)
    generator = Reshape(in_shape)(generator)
    generator_output = ActivityRegularization(l1=1e-5, l2=0.0)(generator)

    if residuals:
        generator_output = Add(name="output")([generator_input, generator_output])

    return Model(inputs=generator_input, outputs=generator_output)


def create_discriminator(in_shape=(28, 28, 1)):
    """Define a neural network binary classifier to classify real and generated
    examples."""

    model = Sequential(
        [
            Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=in_shape),
            GaussianNoise(0.2),
            LeakyReLU(negative_slope=0.2),
            Dropout(0.4),
            Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            LeakyReLU(negative_slope=0.2),
            Dropout(0.4),
            Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            LeakyReLU(negative_slope=0.2),
            Dropout(0.4),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def define_countergan(generator, discriminator, classifier, input_shape=(28, 28, 1)):
    """Combine a generator, discriminator, and fixed classifier into the CounteRGAN."""
    discriminator.trainable = False
    classifier.trainable = False

    countergan_input = Input(shape=input_shape, name="countergan_input")

    x_generated = generator(countergan_input)

    countergan = Model(
        inputs=countergan_input,
        outputs=[discriminator(x_generated), classifier(x_generated)],
    )

    optimizer = optimizers.RMSprop(learning_rate=4e-4, decay=1e-8)  # Generator optimizer
    countergan.compile(
        optimizer, loss=["binary_crossentropy", "categorical_crossentropy"]
    )
    return countergan
