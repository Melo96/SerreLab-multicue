import numpy as np
import pandas as pd
import os
import pickle
import random
from tqdm import tqdm
from tqdm import trange
#from matplotlib.image import imread
from skimage.io import imread
import gc
import pyarrow as pa

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, InputLayer
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
from keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import tempfile
import multiprocessing
import threading
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import pretrained ResNet50
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import pdb

# image size and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224 #image size
bs = 128 #batch size

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Seed value
# Apparently you may use different seed values at each stage
i = 1
seed_value= 42 * i

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
# tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Save Path
save_path = "/home/zshen15/multi_cue/coordinates/"

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
print('Number of CPUs: {}'.format(multiprocessing.cpu_count()))

# read the dataframe, add a normalized column and a binary column
datadf = pd.read_parquet('/media/data_cifs/projects/prj_multi-cue/zhujun_texture/model/datadf.parquet.gzip')

l = datadf['distance']
mu = np.mean(l)
sd = np.std(l)

def norm_distance(ori_distance, mu = mu, sd = sd):
	return (ori_distance - mu) / sd

datadf['normalized_distance'] = datadf.distance.apply(norm_distance)

def binary_distance(ori_distance, mu = mu):
	if ori_distance >= mu:
		return '1'
	else:
		return '0'
datadf['binary_distance'] = datadf.distance.apply(binary_distance)

n = int(len(datadf))
#p = 0.02
#datadf = datadf[:int(p*n)]
datadf = datadf[:int(0.8*n)]

n = int(len(datadf))

STEP_SIZE_TRAIN=int(0.75 * n//bs + 1)
STEP_SIZE_VALID=int(0.25 * n//bs + 1)
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

traindf = datadf.iloc[:int(0.75*n),:]
validdf = datadf.iloc[int(0.75*n):,:]

all_label = datadf['binary_distance'].to_numpy()
all_label = tf.keras.utils.to_categorical(all_label,2)

train_label = all_label[:int(0.75*n)]
valid_label = all_label[int(0.75*n):]

print('Reading img array')
train_img = np.empty((len(traindf), 224, 224, 3))
valid_img = np.empty((len(validdf), 224, 224, 3))

tqdm.pandas()
i = 0
for path in tqdm(traindf['path'].values):
	train_img[i,:,:,:] = imread(path)
	i+=1

i = 0
for path in tqdm(validdf['path'].values):
	valid_img[i,:,:,:] = imread(path)
	i+=1

#del train_table
#del valid_table
gc.collect()

def add_regularization(model, regularizer = tf.keras.regularizers.l1(8e-5)):

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

# featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, vertical_flip=True, 
datagen=ImageDataGenerator()

print('Generating training data')
train_ds=datagen.flow(train_img, train_label,  batch_size=bs, shuffle=True)

# train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_label))
print('Done')
# train_ds = train_ds.shuffle(5000).batch(bs).prefetch(tf.data.experimental.AUTOTUNE).repeat()

print('Generating validation data')
valid_ds=datagen.flow(valid_img, valid_label,  batch_size=bs, shuffle=False)

#valid_ds = tf.data.Dataset.from_tensor_slices((valid_img, valid_label))
print('Done')
# valid_ds = valid_ds.shuffle(5000).batch(bs).prefetch(tf.data.experimental.AUTOTUNE).repeat()

print('Done')

# Pretrained Resnet function

with strategy.scope():

	restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
	output = restnet.layers[-1].output
	output = keras.layers.GlobalAveragePooling2D()(output)
	restnet = Model(restnet.input, outputs=output)

	# for layer in restnet.layers:
	#    layer.trainable = False

	restnet = add_regularization(restnet)

	model = Sequential()
	model.add(restnet)
	# model.add(Dense(512, activation='elu', activity_regularizer=regularizers.l1(1e-4), input_dim=(IMG_HEIGHT, IMG_WIDTH, 3)))
	model.add(Dropout(0.5))
	# model.add(Dense(512, activation='elu', activity_regularizer=regularizers.l1(1e-4)))
	# model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

class ReportValidationStatus(tf.keras.callbacks.Callback):

        def on_test_batch_begin(self, batch, logs=None):
            print('Evaluating: batch {} begins at {}'.format(batch, datetime.now().time()))

        def on_test_batch_end(self, batch, logs=None):
            print('Evaluating: batch {} ends at {}'.format(batch, datetime.now().time()))

csv_logger = CSVLogger("stage1_l1_8e-5_log.csv", append=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  factor=0.5, patience=5,verbose=1)
# earlystop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('l1_8e-5.{epoch:02d}-{val_loss:.4f}.hdf5', monitor = 'val_loss',
                                 verbose = 0, save_best_only=True, period=5, 
                                 mode = 'min', save_weights_only=True)

# fit model
# model.load_weights('100_.20-1.0581.hdf5')

# lr_scheduler, checkpoint, csv_logger

gc.collect()

history = model.fit(train_ds, 
        steps_per_epoch=STEP_SIZE_TRAIN, use_multiprocessing=False, workers=1, 
    	validation_data=valid_ds, 
        validation_steps=STEP_SIZE_VALID,
    	callbacks=[lr_scheduler, checkpoint, csv_logger],
    	epochs=30)

model.save('resnet50_50.h5')

# Y_pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
# pred_df = pd.DataFrame(Y_pred)
# pred_df.to_csv('test_predict.csv')
# model.evaluate(test_generator,steps=STEP_SIZE_TEST)