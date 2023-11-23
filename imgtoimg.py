import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose

# Define the Generator (U-Net architecture)
def build_generator(input_shape, output_channels):
    inputs = Input(shape=input_shape)
    
    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
    ]

    up_stack = [
        upsample(256, 4),  # (batch_size, 64, 64, 256)
        upsample(128, 4),  # (batch_size, 128, 128, 128)
        upsample(64, 4),  # (batch_size, 256, 256, 64)
    ]

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = keras.layers.Concatenate()([x, skip])

    # Output layer
    output = Conv2DTranspose(output_channels, 4, strides=2, padding='same', activation='tanh')(x)  # (batch_size, 256, 256, 3)

    return keras.Model(inputs=inputs, outputs=output)

# Define the Discriminator
def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    
    x = inputs
    x = downsample(64, 4, apply_batchnorm=False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=x)

# Utility function to downsample
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(Activation('relu'))
    return result

# Utility function to upsample
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

# Define the input shape and number of output channels (e.g., for color images, it's 3)
input_shape = (256, 256, 3)
output_channels = 3  # RGB

# Build the generator and discriminator
generator = build_generator(input_shape, output_channels)
discriminator = build_discriminator(input_shape)

# Print model summaries
generator.summary()
discriminator.summary()
 
