# MODEL TRAINING SCRIPT
import logging
import os

# third party libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1.5GB of memory on the GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1500)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# # or set memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print('Found GPUs:', gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, enable=True)

# custom libraries
from src.data import data_loading
from src.models import model as src_model


# set logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
# logger = logging.getLogger('training')

# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
#
# ch.setFormatter(CustomFormatter())
#
# logger.addHandler(ch)


# add callbacks (model_checkpoint, tensorboard)
# define callbacks
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


# global custom variables
DATA_DIR = '../data'
IMG_SIZE = 256, 256


# PREPARE DATA FOR TRAINING

# load the training_df.csv
df = pd.read_csv(os.path.join(DATA_DIR, 'training_df.csv'))

# partition into train, validation and test, stratify by artist
train_df, val_test_df = train_test_split(df,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=df['artist'])
validation_df, test_df = train_test_split(val_test_df,
                                          test_size=0.5,
                                          random_state=42,
                                          stratify=val_test_df['artist'])
# shuffle the dataframes
train_df = train_df.sample(frac=1, random_state=42)
validation_df = validation_df.sample(frac=1, random_state=42)
test_df = test_df.sample(frac=1, random_state=42)

# simple info on data distribution
print(f'Number of train pictures: {len(train_df)}')
print(f'Number of validation pictures: {len(validation_df)}')
print(f'Number of test pictures: {len(test_df)}')

# apply data streaming pipelines
train_ds = data_loading.input_pipeline(train_df, img_size=IMG_SIZE)
train_ds = data_loading.performance_pipeline(train_ds, batchsize=32, bufsiz=32)
validation_ds = data_loading.input_pipeline(validation_df, img_size=IMG_SIZE)
validation_ds = data_loading.performance_pipeline(validation_ds, batchsize=32)
test_ds = data_loading.input_pipeline(test_df, img_size=IMG_SIZE)
test_ds = data_loading.performance_pipeline(test_ds)

# DEFINE MODEL AND TRAIN

num_classes = len(df['id'].unique())

model = src_model.get_model(num_classes=num_classes, input_shape=(*IMG_SIZE, 3))

# define optimizer, loss, and metrics
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
loss = keras.losses.categorical_crossentropy
metrics = [
    keras.metrics.CategoricalAccuracy(),
]

# compile the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics
              )

# print model summary
model.summary()


lr_schdl = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

history = model.fit(train_ds,
                    validation_data=validation_ds,
                    epochs=15,
                    callbacks=[lr_schdl]
                    )
