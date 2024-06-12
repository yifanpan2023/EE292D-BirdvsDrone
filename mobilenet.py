import tempfile
import os

import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
print(tf.sysconfig.get_build_info())
print(tf.config.list_physical_devices('GPU'))

import os
import shutil
import numpy as np
from tensorflow_model_optimization.python.core.keras.compat import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras import layers
import keras

tf.random.set_seed(
    666
)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int)
args = parser.parse_args()
print(args.layer)

import math
n_batches = (4548) / 32
n_batches = math.ceil(n_batches)

os.makedirs("mvb_new_model_test/low_training_"+str(args.layer), exist_ok=True)
checkpoint_path = "mvb_new_model_test/low_training_"+str(args.layer)+"/best.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)



cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True,
    verbose=0)

train_ds,valid_ds = tf.keras.utils.image_dataset_from_directory(
  "./mvb_dataset_test/train/", label_mode = 'categorical', image_size =(128, 128),
  validation_split = 0.1, subset = 'both', batch_size = 32, seed = 111)




data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomContrast(0.2)
])

aug_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

base_model =keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,
                                           weights='imagenet')
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = args.layer

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

x =  keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(3)(x)

model = keras.Model(base_model.input, x)

model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=60,
  batch_size=32,
  callbacks=[cp_callback]
  
)

"""
100: 22, 11
120:18
80: 13, 8, 6
60: 30, 29, 23, 8
40: 24, 25, 30, 4, 5, 13, 18, 19
20: 19, 20, 23, 25, 
"""

