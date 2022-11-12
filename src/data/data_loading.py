# FUNCTIONS AND UTILITIES FOR DATA LOADING
import logging
import os

import matplotlib.pyplot as plt
# third party libraries
import pandas as pd
import tensorflow as tf

# set logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# global variables
AUTOTUNE = tf.data.AUTOTUNE


def decode_image(filepath, img_size):
    image = tf.io.decode_jpeg(tf.io.read_file(filepath))
    # TODO: THINK ABOUT A BETTER WAY TO RESIZE
    # TODO: VARIOUS CROPPINGS: HOW?
    if img_size:
        image = tf.image.resize(image, img_size)  # resize to uniform size
    image = image/255.  # rescale image in [0,1]
    return image


def input_pipeline(dataframe: pd.DataFrame, img_size=None):

    num_classes = len(dataframe['target'].unique())

    def process_row(row):
        label_oh = tf.one_hot(row['target'],
                              depth=num_classes)
        image = decode_image(row['filepath'], img_size=img_size)

        # there are some weird grayscale images (H,W,None) or (H,W,1)
        if img_size:
            if tf.shape(image)[-1] != 3:
                if tf.rank(image) == 2:
                    image = tf.stack([image] * 3, axis=2)
                elif tf.shape(image)[-1] == 1:
                    image = tf.repeat(image, 3, axis=2)
                image = tf.reshape(image, (*img_size, 3), )

        return image, label_oh

    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    ds = ds.map(process_row, num_parallel_calls=AUTOTUNE)
    return ds


def performance_pipeline(ds: tf.data.Dataset, batchsize=0, bufsiz=0, cache=False, cache_path: str = ""):
    if cache:
        ds = ds.cache(cache_path)
    if bufsiz > 0:
        ds = ds.shuffle(buffer_size=bufsiz)
    if batchsize > 0:
        ds = ds.batch(batchsize)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == '__main__':
    from src.data import df_utils

    DATA_DIR = '../../data'

    img_dir = os.path.join(DATA_DIR, 'images')
    df = df_utils.create_dataframe(img_dir)

    # add dummy target label
    df['target'] = [i % 7 for i in range(len(df))]

    # apply pipelines
    ds = input_pipeline(df, img_size=(256, 256))
    ds = performance_pipeline(ds, batchsize=16, bufsiz=16)

    print('Checking one element in dataset')
    for elem in ds.take(1):
        print('data_shape:', elem[0].shape)
        print('target_shape:', elem[1].shape)
        plt.imshow(elem[0][0])
        plt.title(f'target: {elem[1][0]}')
        plt.show()
