""" Model architectures and utilities """

# third party libraries
from tensorflow import keras
from keras import layers


# define data preprocessing pipeline
def data_preprocessing():
    """Defines the preprocessing pipeline"""

    _data_preprocessing = keras.Sequential([
        layers.Rescaling(1./255)
    ])
    return _data_preprocessing


# define data augmentation pipeline
def data_augmentation():
    """Defines the augmentation pipeline"""

    _data_augmentation = keras.Sequential([
        layers.RandomFlip(mode='horizontal_and_vertical'),
        # layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ])
    return _data_augmentation


# here we define a custom model from scratch
def get_model(num_classes: int, input_shape: tuple[int, int, int]):
    """
    Return a fresh randomly initialized model

    Args:
        num_classes (int): number of classes
        input_shape (tuple[int, int, int]): shape of the input
    """

    # define input shape
    inputs = layers.Input(input_shape)

    # apply preprocessing
    x = data_preprocessing()(inputs)

    # apply augmentation
    x = data_augmentation()(x)

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

    output = layers.Dense(num_classes, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=output)


def from_pretrained_model(base_model: keras.models.Model, num_classes: int, input_shape: tuple[int], preproc_func=None):
    """
    Return a model using a pretrained backbone

    Args:
        base_model (keras.models.Model): base model to be used as backbone
        num_classes (int): number of classes
        input_shape (tuple[int, int, int]): shape of the input
    """

    # define input shape
    inputs = layers.Input(input_shape)

    # apply preprocessing
    if preproc_func:
        processed = preproc_func(inputs)
    else:
        processed = inputs

    # apply augmentation
    augmented = data_augmentation()(processed)

    # pass to base model
    x = base_model(augmented)

    # Add layers at the end
    x = layers.Flatten()(x)

    x = layers.Dense(512, kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(16, kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=output)
