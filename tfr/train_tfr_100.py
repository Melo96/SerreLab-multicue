import numpy as np
import pandas as pd
import os
import pickle
import random
from tqdm import tqdm
from tqdm import trange
import gc
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from PIL import Image
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, InputLayer
from keras import regularizers, optimizers
from keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import tempfile
import multiprocessing
import threading
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# import pretrained ResNet50
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import pdb

# image size and batch size
img_size = 224 #image size
bs = 20 #batch size
seed = 42
AUTO = tf.data.experimental.AUTOTUNE

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
#os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Save Path
save_path = "/media/data_cifs/projects/prj_multi-cue/coordinates/all_data_tfr"

strategy = tf.distribute.MirroredStrategy()
print('Number of GPUs: {}'.format(strategy.num_replicas_in_sync))
print('Number of CPUs: {}'.format(multiprocessing.cpu_count()))

# read the dataframe, add a normalized column and a binary column
# df = pd.read_parquet(save_path + '/df_100.parquet', columns=['x', 'y', 'binary_distance', 'path'])
# df = df.iloc[:1000]

# paths = df['path']
# xys = df[['x', 'y']]
# labels = df['binary_distance']

filenames = [save_path + "/train_19.tfrecords", save_path + "/train_18.tfrecords",
		save_path + "/train_17.tfrecords", save_path + "/train_16.tfrecords",
]

def decode_image(img):
	img = tf.image.decode_png(img, channels=3)
	img = tf.cast(img, tf.float32) / 255.0
	return img

def read_tfrecord(example):
	LABELED_TFREC_FORMAT = {
          			"image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
          			"xy": tf.io.FixedLenFeature([], tf.string), 
          			"label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    		}
	example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
	image = decode_image(example['image'])
	label = tf.cast(example['label'], tf.int64)
	xy = tf.io.decode_raw(example['xy'], tf.float64)
	return (image, xy), label

def build_augmenter(input, label):
	img = tf.image.random_flip_left_right(input[0])
	img = tf.image.random_flip_up_down(img)
	img = tf.image.random_saturation(img, 0.8, 1.2)
	img = tf.image.random_brightness(img, 0.2)
	img = tf.image.random_contrast(img, 0.8, 1.2)
	img = tf.image.random_hue(img, 0.2)
	return (img, input[1]), label

def build_ds(files, aug=True, repeat=True, shuffle=True):
	ds = tf.data.TFRecordDataset(filenames = files)
	ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
	ds = ds.map(build_augmenter, num_parallel_calls=AUTO) if aug else ds
	ds = ds.repeat() if repeat else ds
	ds = ds.shuffle(100) if shuffle else ds
	ds = ds.batch(bs).prefetch(AUTO)
	return ds

def add_regularization(model, regularizer = tf.keras.regularizers.l1(1e-4)):

	if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
		print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
		return model

	for layer in model.layers:
		for attr in ['kernel_regularizer']:
			if hasattr(layer, attr):
				setattr(layer, attr, regularizer)

	# When we change the layers attributes, the change only happens in the model config file
	model_json = model.to_json()

	# Save the weights before reloading the model.
	tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
	model.save_weights(tmp_weights_path)

	# load the model from the config
	model = tf.keras.models.model_from_json(model_json)

	# Reload the model weights
	model.load_weights(tmp_weights_path, by_name=True)
	return model

def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)

print('Done')

# Pretrained Resnet function

def build_model():
	keras.backend.clear_session()
	np.random.seed(seed)
	tf.random.set_seed(seed)

	with strategy.scope():

		resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size,img_size,3))
		for layer in resnet.layers:
			layer.trainable = True

		output = resnet.layers[-1].output
		output = keras.layers.GlobalAveragePooling2D()(output)
		# output = keras.layers.Flatten()(output)
		resnet = Model(resnet.input, outputs=output)

		# resnet = add_regularization(resnet)

		input2 = keras.layers.Input(shape=(2,))
		model = Dense(128, activation='relu')(input2)
		model = BatchNormalization()(model)
		model = Dropout(0.4)(model)
		model = Dense(64, activation='relu')(model)
		model2 = Model(input2, outputs=model)

		model = keras.layers.Concatenate(axis=1)([model2.output, resnet.output])
		model = Dense(128, activation='relu')(model)
		model = BatchNormalization()(model)
		model = Dropout(0.4)(model)
		output = Dense(1, activation='sigmoid')(model)
		model = keras.models.Model(inputs=[resnet.input, input2], outputs=output)

		# model = add_regularization(model)

		model.compile(loss='binary_crossentropy',
             		 optimizer=optimizers.Adam(lr=1.6000001e-06),
              		  metrics=['accuracy'])
	return model

class ReportValidationStatus(tf.keras.callbacks.Callback):

        def on_test_batch_begin(self, batch, logs=None):
            print('Evaluating: batch {} begins at {}'.format(batch, datetime.now().time()))

        def on_test_batch_end(self, batch, logs=None):
            print('Evaluating: batch {} ends at {}'.format(batch, datetime.now().time()))

# fit model
# model.load_weights('mar_tfr_100_fold0.hdf5')

# K folds
keras.backend.clear_session()
np.random.seed(seed)
tf.random.set_seed(seed)

TRAINING_FILENAMES =  tf.io.gfile.glob(save_path + '/*.tfrecords')
TRAINING_FILENAMES = TRAINING_FILENAMES[10:13]

save_path = "/media/data_cifs/projects/prj_multi-cue/coordinates/100data/result_100_mar"

folds = 3

kfold = KFold(3, shuffle=True, random_state=seed)

for f, (train_index, val_index) in enumerate(kfold.split(TRAINING_FILENAMES)):
	print("training fold {}".format(f))
	train_files = list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[train_index]['TRAINING_FILENAMES'])
	valid_files = list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_index]['TRAINING_FILENAMES'])
	train_ds = build_ds(train_files, aug=False)
	valid_ds = build_ds(valid_files, aug=False, repeat=False, shuffle=False)

	model = build_model()
	model.load_weights('mar_tfr_100_fold0.hdf5')

	# Callbacks
	lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  factor=0.2, patience=2, verbose=1, min_lr=1e-6)
	checkpoint = keras.callbacks.ModelCheckpoint(save_path + "/mar_tfr_100_fold{}.hdf5".format(f), monitor = 'val_loss',
	                                 verbose = 0, save_best_only=True,
	                                 mode = 'min', save_weights_only=True)
	earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
	csv_logger = keras.callbacks.CSVLogger(save_path + "/mar_tfr_100_fold{}.csv".format(f), append=True)

	NUM_TRAINING_IMAGES = int(81842 *2)
	NUM_VALIDATION_IMAGES = int(81842)
	# print(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES)

	model.fit(train_ds, validation_data=valid_ds, epochs=100, verbose=1, 
			steps_per_epoch = NUM_TRAINING_IMAGES // bs + 1, initial_epoch=27, 
          		callbacks=[lr_scheduler, csv_logger, checkpoint]
            		)