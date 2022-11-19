# MODEL TRAINING SCRIPT
import logging
import datetime
import os
import time

# third party libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# custom libraries
from src.data import data_loading
from src.models import model as src_model


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


# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


# global custom variables
DATA_DIR: str = '../data'
IMG_SIZE: tuple[int, int] = 256, 256


# PREPARE DATA FOR TRAINING

# load the training, validation and test datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
validation_df = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

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
train_ds = data_loading.performance_pipeline(train_ds, batchsize=32, shuffle_bufsiz=32)
validation_ds = data_loading.input_pipeline(validation_df, img_size=IMG_SIZE)
validation_ds = data_loading.performance_pipeline(validation_ds, batchsize=32)
test_ds = data_loading.input_pipeline(test_df, img_size=IMG_SIZE)
test_ds = data_loading.performance_pipeline(test_ds)

# compute class weights
print(pd.concat([train_df, validation_df, test_df]).groupby('id').count())
print(pd.concat([train_df, validation_df, test_df]).groupby('id').max())

counts = pd.concat([train_df, validation_df, test_df]).groupby('id').count()['filepath'].to_dict()
total = sum(counts.values())
class_weights_dict = {}
for ids, count in counts.items():
    class_weights_dict.update({ids: total/(len(counts)*count)})
# TODO: smoothing?

# DEFINE MODEL AND TRAIN ONLY TOP LAYERS

unique_ids = pd.concat([train_df, validation_df, test_df])['id'].unique()
num_classes = len(unique_ids)

# Load pre-trained model
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
# freeze the base model
for layer in base_model.layers:
    layer.trainable = False

model = src_model.from_pretrained_model(base_model, num_classes=num_classes, input_shape=(*IMG_SIZE, 3))

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

print('Model summary with backbone frozen layers:')
# print model summary
model.summary()

# define callbacks
ts = time.time()
timestring = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
model_folder = os.path.join('../models/temp/', timestring)
model_saver = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(model_folder, 'epoch{epoch:02d}_vl{val_loss:.2f}_va{val_categorical_accuracy}.h5'),
        )
tb_callback = tf.keras.callbacks.TensorBoard(os.path.join(model_folder, 'logs'), profile_batch=10)
lr_schdl = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# start training
history1 = model.fit(train_ds,
                     validation_data=validation_ds,
                     epochs=20,
                     callbacks=[lr_schdl,
                                model_saver,
                                tb_callback]
                     )

# NOW FINE TUNE UPPER LAYERS OF BASE MODEL

# unfreeze last layers
for layer in model.layers:
    if 'conv5' in layer.name:
        layer.trainable = True

# redefine optimizer, loss, and metrics
optimizer = keras.optimizers.Adam(learning_rate=1e-6)
loss = keras.losses.categorical_crossentropy
metrics = [
    keras.metrics.CategoricalAccuracy(),
]

# recompile the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics
              )

print('Model summary with some trainable backbone layers:')
# print model summary
model.summary()

# continue training
history2 = model.fit(train_ds,
                     validation_data=validation_ds,
                     epochs=20,
                     class_weight=class_weights_dict,
                     callbacks=[lr_schdl,
                                model_saver,
                                tb_callback]
                     )
