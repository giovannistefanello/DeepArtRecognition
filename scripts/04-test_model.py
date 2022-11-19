# MODEL TESTING SCRIPT
import logging
import datetime
import os
import time

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklmetrics
import tensorflow as tf
from tensorflow import keras

# custom libraries
from src.data import data_loading


# or set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Found GPUs:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)


# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# global custom variables
DATA_DIR: str = '../data'
MODEL_DIR: str = '../models'
IMG_SIZE: tuple[int, int] = 256, 256


# PREPARE DATA FOR TESTING

# load the train, validation and test datasets
# load the training, validation and test datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
validation_df = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# simple info on data
print(f'Number of test pictures: {len(test_df)}')
df = pd.concat([train_df, validation_df, test_df])
uniques = df[['artist', 'id']].drop_duplicates()
id_to_artist = dict(zip(uniques['id'], uniques['artist']))

# apply data streaming pipeline
test_ds = data_loading.input_pipeline(test_df, img_size=IMG_SIZE)
test_ds = data_loading.performance_pipeline(test_ds, batchsize=1)

# Load pre-trained model
model = keras.models.load_model(os.path.join(MODEL_DIR, 'model.h5'))

# Test model
preds = model.predict(test_ds)
guesses = np.argmax(preds, axis=-1)
guesses_cat = [id_to_artist[i] for i in guesses]
true = test_df['id'].to_numpy()
true_cat = [id_to_artist[i] for i in true]
# compute and show confusion matrix
conf_mat = sklmetrics.confusion_matrix(true_cat, guesses_cat)
fig, ax = plt.subplots(figsize=(17, 17))
sklmetrics.ConfusionMatrixDisplay.from_predictions(true_cat, guesses_cat, normalize='true', ax=ax)
plt.show()
