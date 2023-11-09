import random
import pathlib

import itertools
import collections
from pathlib import Path

import os
import cv2
import numpy as np
import remotezip as rz
import shutil
import zipfile
import tqdm
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

subset_paths = {}
subset_paths['train'] = Path('./Data/train')
subset_paths['test'] = Path('./Data/test')
subset_paths['val'] = Path('./Data/val')


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """
      Creates frames from each video file present for each category.

      Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

      Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(
            set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx)
                                       for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]
              # Encode labels
            yield video_frames, label

# fg = FrameGenerator(subset_paths['train'], 6, training=True)
# frames, label = next(fg())
# print(f"Shape: {frames.shape}")
# print(f"Label: {label}")
# Create the training set
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),                    tf.TensorSpec(shape=(), dtype=tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 6, training=True),                                        output_signature=output_signature)



# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 6),
                                        output_signature=output_signature)

# create the test set
test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], 6),
                                         output_signature=output_signature)


# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)
test_ds = val_ds.batch(2)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

test_frames, test_labels = next(iter(test_ds))
print(f'Shape of test set of frames: {test_frames.shape}')
print(f'Shape of test labels: {test_labels.shape}')


net = tf.keras.applications.EfficientNetB0(include_top=False)
net.trainable = False

# adapted from https://www.tensorflow.org/guide/keras/transfer_learning
# Define your model as a function
def build_model(hp):
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),
    tf.keras.layers.TimeDistributed(net),
    tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
    tf.keras.layers.GlobalAveragePooling3D()
    ])
    

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        beta_1=hp.Choice('beta_1', values=[0.9, 0.99, 0.999]),
        beta_2=hp.Choice('beta_2', values=[0.999, 0.9999]),
        epsilon=hp.Float('epsilon', min_value=1e-10, max_value=1e-7)
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Adjust the number of trials as needed
    directory='my_tuning_directory'  # Choose a directory for saving tuning results
)

# bayesian

# Define callbacks, such as early stopping, if necessary
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]

# Start the tuning process
tuner.search(x=train_ds, epochs=20, validation_data=(val_ds), callbacks=callbacks)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hps)

# Build and compile the final model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
best_model.compile(optimizer=best_hps['optimizer'], loss='binary_crossentropy', metrics=['recall'])
# using metrics : precison, recall, f1. use f1
# Train the final model
best_model.fit(x=train_ds, epochs=10, validation_data=(val_ds, val_labels))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=10,
          validation_data=val_ds,
          callbacks=tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'))

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], 6),
                                         output_signature=output_signature)

test_ds = test_ds.batch(2)

test_loss, test_accuracy = model.evaluate(test_ds)

model.save('gfgModel.h5')

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print(model.summary())
