import os
import pandas as pd
import pickle
import pdb
from tqdm import tqdm
from tqdm import trange
import numpy as np
from matplotlib.image import imread
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from PIL import Image
import tensorflow as tf
import gc

# build a dataframe with 'image_id', 'distance', 'path'
# datadf = pd.DataFrame(columns=['image_id', 'distance', 'path'])
# os.chdir('/media/data_cifs/projects/prj_multi-cue/rendered_datasets/images_texture_only_cycles/')

height, width = 224, 224

df = pd.read_parquet('./data_1000_validated.parquet.gzip')

# ftrs = ['x', 'y', 'binary_distance', 'normalized_distance']
n = int(len(df) // 20)
tqdm.pandas()

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

dict = {}

for i in range(10, 20):
	print('Processing the {}th TF Record'.format(i))
	filename = '/media/data_cifs/projects/prj_multi-cue/coordinates/all_data_tfr/train_{}.tfrecords'.format(i)
	writer = tf.io.TFRecordWriter(filename)

	cur_df = df.iloc[i*n:(i+1)*n, :]
	cnt = 0
	
	for j in trange(len(cur_df)):
		cur_path = cur_df['path'].values[j]
		img = tf.io.read_file(cur_path)
		xy = cur_df[['x', 'y']].values[j]
		label = int(cur_df['binary_distance'].values[j])

		feature = {
			'image': _bytes_feature(img.numpy()),
    			'xy': _bytes_feature(xy.tobytes()),
    			'label': _int64_feature(label),
		}	
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(example.SerializeToString())
		cnt+=1

	writer.close()
	dict[i] = [cnt]

os.chdir('/media/data_cifs/projects/prj_multi-cue/coordinates/all_data_tfr')
df = pd.DataFrame.from_dict(dict)
df.to_csv("tfr_nums_10-19.csv")
print('Success!')
