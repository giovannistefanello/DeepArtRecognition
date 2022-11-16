# MODEL ARCHITECTURE AND UTILITIES

# third party libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers


# define data preprocessing pipeline
data_preprocessing = keras.Sequential([
    layers.Rescaling(1./255)
])

# define data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip(mode='horizontal_and_vertical'),
    # layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
])


# here we define a custom model from scratch
def get_model(num_classes: int, input_shape: tuple[int, int, int]):
    # define input shape
    inputs = layers.Input(input_shape)

    # apply preprocessing
    x = data_preprocessing(inputs)

    # apply augmentation
    x = data_augmentation(x)

    # feature extractor and predictor
    x = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('leaky_relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def from_pretrained_model(base_model: keras.models.Model, num_classes: int, input_shape: tuple[int]):
    # TODO: finish function with transfer learning model setup

    return
