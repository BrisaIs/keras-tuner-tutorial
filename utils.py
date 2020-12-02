import tensorflow as tf
from loguru import logger
import os
import pathlib
import numpy as np
batch_size = 32
img_height = 350
img_width = 350

def set_gpu_config():
    # Set up GPU config
    logger.info("Setting up GPU if found")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def load_data():
    training_data_dir = "/scratch/201903073c-1/nudity/training"
    training_data_dir = pathlib.Path(training_data_dir)
    training_image_count = len(list(data_dir.glob('*/*.jpg')))

    testing_data_dir = "/scratch/201903073c-1/nudity/testing"
    testing_data_dir = pathlib.Path(testing_data_dir)
    testing_image_count = len(list(data_dir.glob('*/*.jpg')))

    training_list_ds = tf.data.Dataset.list_files(str(training_data_dir/'*/*'), shuffle=False)
    training_list_ds = list_ds.shuffle(training_image_count, reshuffle_each_iteration=False)

    testing_list_ds = tf.data.Dataset.list_files(str(testing_data_dir/'*/*'), shuffle=False)
    testing_list_ds = list_ds.shuffle(testing_image_count, reshuffle_each_iteration=False)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for image, label in train_ds:
        x_train.add(image)
        y_train.add(label)

    for image, label in val_ds:
        x_test.add(image)
        y_test.add(label)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_test, x_train, y_test, y_train
