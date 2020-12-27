import numpy as np
import pandas as pd
import os
import pickle
import random
from tqdm import tqdm
from tqdm import trange
from matplotlib.image import imread
import gc

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
IMG_HEIGHT, IMG_WIDTH = 64, 64 #image size
bs = 128 #batch size

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

def add_regularization(model, regularizer = tf.keras.regularizers.l2(2e-4)):

	if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
		print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
		return model

	for layer in model.layers:
		if layer.name[-4:] == 'conv':
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

# shuffle it and divide it into train and test set
# datadf = datadf.sample(frac=1, random_state=seed)

# featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, vertical_flip=True, 
# datagen=ImageDataGenerator()

def traingen_multi_inputs(traindf,train_data, train_label, bs):
	train_img = tf.data.Dataset.from_tensor_slices(train_data)
	train_y = tf.data.Dataset.from_tensor_slices(train_label)
	train_xy = tf.data.Dataset.from_tensor_slices(traindf[['x','y']].values.reshape(traindf.shape[0],2))
	train_img = train_img.batch(bs,drop_remainder=True).repeat()
	train_y = train_y.batch(bs,drop_remainder=True).repeat()
	train_xy = train_xy.batch(bs,drop_remainder=True).repeat()
	train_img = train_img.as_numpy_iterator()
	train_y = train_y.as_numpy_iterator()
	train_xy = train_xy.as_numpy_iterator()
	while True:
		X1i = train_img.next()
		X1j = train_y.next()
		X2i = train_xy.next()
		yield [X1i, X2i], X1j

def validgen_multi_inputs(validdf,valid_data, valid_label, bs):
	valid_img = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_y = tf.data.Dataset.from_tensor_slices(valid_label)
	valid_xy = tf.data.Dataset.from_tensor_slices(validdf[['x','y']].values.reshape(validdf.shape[0],2))
	valid_img = valid_img.batch(bs,drop_remainder=True).repeat()
	valid_y = valid_y.batch(bs,drop_remainder=True).repeat()
	valid_xy = valid_xy.batch(bs,drop_remainder=True).repeat()
	valid_img = valid_img.as_numpy_iterator()
	valid_y = valid_y.as_numpy_iterator()
	valid_xy = valid_xy.as_numpy_iterator()
	while True:
		X1i = valid_img.next()
		X1j = valid_y.next()
		X2i = valid_xy.next()
		yield [X1i, X2i], X1j

print('Done')

# Pretrained Resnet function

with strategy.scope():

	# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
	resnet = ResNet50(include_top=False, weights=None, input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
	output = resnet.layers[-1].output
	output = keras.layers.GlobalAveragePooling2D()(output)
	resnet = Model(resnet.input, outputs=output)

	#for layer in resnet.layers:
	#	layer.trainable = False

	#resnet = add_regularization(resnet)

	input2 = keras.layers.Input(shape=(2,))

	# , kernel_regularizer=tf.keras.regularizers.l1(1e-4)
	model = Dense(128, activation='relu')(input2)
	model = BatchNormalization()(model)
	model = Dropout(0.4)(model)
	model = Dense(64, activation='relu')(model)
	model = BatchNormalization()(model)
	# model = Dropout(0.3)(model)
	model2 = Model(input2, outputs=model)

	model = keras.layers.Concatenate(axis=1)([model2.output, resnet.output])
	# model = Dense(512, activation='relu')(model)
	# model = Dropout(0.2)(model)
	# model = Dense(256, activation='relu')(model)
	model = Dropout(0.4)(model)
	model = Dense(128, activation='relu')(model)
	model = BatchNormalization()(model)
	# model = Dropout(0.2)(model)
	# model = Dense(64, activation='relu')(model)
	model = Dropout(0.4)(model)
	output = Dense(2, activation='softmax')(model)
	model = keras.models.Model(inputs=[resnet.input, input2], outputs=output)

	model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

class ReportValidationStatus(tf.keras.callbacks.Callback):

        def on_test_batch_begin(self, batch, logs=None):
            print('Evaluating: batch {} begins at {}'.format(batch, datetime.now().time()))

        def on_test_batch_end(self, batch, logs=None):
            print('Evaluating: batch {} ends at {}'.format(batch, datetime.now().time()))

csv_logger = CSVLogger("alldata_noweight_epoch10_log.csv", append=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  factor=0.5, patience=10,verbose=1)
# earlystop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('alldata_noweight.{epoch:02d}-{val_loss:.4f}.hdf5', monitor = 'val_loss',
                                 verbose = 0, save_best_only=False, period=1, 
                                 mode = 'min', save_weights_only=True)

# model.load_weights('alldata_run1.01-0.3933.hdf5')

# read the dataframe, add a normalized column and a binary column
for i in range(10):
	print('{} th training loop'.format(i))
	datadf = pd.read_parquet('./resized_data_parquet/data_img_{}.parquet'.format(i), columns=['x', 'y', 'binary_distance'])

	n = int(len(datadf))

	# generating training set and change it to dataframiterator
	traindf = datadf.iloc[:int(0.8*n),:]
	validdf = datadf.iloc[int(0.8*n):,:]

	#p = 0.1
	#datadf = datadf[:int(p*n)]
	#n = int(len(datadf))

	STEP_SIZE_TRAIN=int(0.8 * n//bs)
	STEP_SIZE_VALID=int(0.2 * n//bs)
	#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

	all_label = datadf['binary_distance'].to_numpy()
	all_label = tf.keras.utils.to_categorical(all_label,2)

	train_label = all_label[:int(0.8*n)]
	valid_label = all_label[int(0.8*n):]

	print('Reshape img array')
	df_img = pd.read_parquet('./resized_data_parquet/data_img_{}.parquet'.format(i), columns=['img_array'])
	all_img = []

	tqdm.pandas()

	for img in tqdm(df_img['img_array'].values):
		img = np.reshape(img, (64,64,3))
		all_img.append(img)

	train_data = np.asarray(all_img[:int(0.8*n)])
	valid_data = np.asarray(all_img[int(0.8*n):])

	del df_img
	del all_img
	gc.collect()

	history = model.fit(traingen_multi_inputs(traindf,train_data, train_label, bs),shuffle=True, 
	        		steps_per_epoch=STEP_SIZE_TRAIN, use_multiprocessing=False, workers=1, 
	            		validation_data=validgen_multi_inputs(validdf,valid_data, valid_label, bs), 
	       			validation_steps=STEP_SIZE_VALID,
#				initial_epoch = 10,
    				callbacks=[lr_scheduler, checkpoint, csv_logger],
    				epochs=10)
	
	del traindf
	del validdf
	del train_data
	del valid_data
	gc.collect()

print('Evaluating the model using test data')
testdf = pd.read_parquet('./resized_data_parquet/data_img_9.parquet', columns=['x', 'y', 'binary_distance'])

n = int(len(testdf))

#p = 0.1
#datadf = datadf[:int(p*n)]
#n = int(len(datadf))

STEP_SIZE_TEST=int(n//bs)

all_label = testdf['binary_distance'].to_numpy()
test_label = tf.keras.utils.to_categorical(all_label,2)

print('Reshape img array')
df_img = pd.read_parquet('./resized_data_parquet/data_img_9.parquet', columns=['img_array'])
all_img = []

tqdm.pandas()

for img in tqdm(df_img['img_array'].values):
	img = np.reshape(img, (64,64,3))
	all_img.append(img)

test_data = np.asarray(all_img)

del df_img
del all_img
gc.collect()

# model.evaluate(test_generator,steps=STEP_SIZE_TEST)

#hist_df = pd.DataFrame(history.history)
# hist_df.to_csv('history.csv')

#model_save = os.path.join(save_path, 'resnet50_50.h5')
#model.save(model_save)

Y_pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
pred_df = pd.DataFrame(Y_pred)
pred_df.to_csv('test_predict.csv')
model.evaluate(test_generator,steps=STEP_SIZE_TEST)