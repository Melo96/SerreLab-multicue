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

# build a dataframe with 'image_id', 'distance', 'path'
datadf = pd.DataFrame(columns=['image_id', 'distance', 'path'])
os.chdir('/media/data_cifs/projects/prj_multi-cue/rendered_datasets/images_texture_only_cycles/')

height, width = 64, 64

x = []
y = []
p = []
distance = []
image_id = []

print('Constructing DataFrame')

for count, folder in tqdm(enumerate(os.listdir())):
#for folder in trange(101,201):
	# construc the df for images in one file
	# file_idx = i
	# pdb.set_trace()

	# find label data from 'data_log.pickle'
	description_path = os.path.join('/media/data_cifs/projects/prj_multi-cue/rendered_datasets/images_texture_only_cycles/', str(folder), 'data_log.pickle')
	file = open(description_path,'rb')
	data = pickle.load(file)

	# def the function for image path
	def img_path_for_df(img_id, file_num = folder, pattern='%04d.png', top='/media/data_cifs/projects/prj_multi-cue/rendered_datasets/images_texture_only_cycles/'):
	    """Return file path, 
	        where there are 100 files all together, and each contains about 2500 png:
	    
	      0000     ../file_num/0000.png
	      0001     ../file_num/0001.png
	    """
	    file_num = str(file_num)
	    #return top + file_num + '/' + pattern % img_num
	    return os.path.join(top, file_num, pattern % img_id)

	#  construct a dataframe with three columns: image_id, distance, path
	tem_key = data.keys()
	for i in tem_key:
		image_id.append(i)
	tem_xy = [j['screen_position'] for j in data.values()]
	for i in tem_xy:
		x.append(i[0])
		y.append(i[1])
	tem_dis = [i['diametric_distance'] for i in data.values()]
	for i in tem_dis:
		distance.append(i)
	tem_path = [img_path_for_df(i) for i in data.keys()]
	for i in tem_path:
		p.append(i)

datadf['image_id'] = image_id
datadf['distance'] = distance
datadf['x'] = x
datadf['y'] = y
datadf['path'] = p

print('Validating files')
tqdm.pandas()
df_len = datadf.shape[0]
path_dic = {}
cur_path = []

for i, row in tqdm(datadf.iterrows()):
	cur_dir = row['path'][:-8]
	cur_pic = row['path'][-8:]
	key = row['path'][-13:-9]
	if key in path_dic:
		cur_path = path_dic[key]
	else:
		cur_path = os.listdir(cur_dir)
		path_dic[key] = cur_path
	if cur_pic not in cur_path:
		datadf.drop(i, axis=0, inplace=True)

#print("constructed data frame for {} files".format(count+1))
datadf.reset_index(inplace=True, drop=True)

l = datadf['distance']
mu = np.mean(l)
sd = np.std(l)

sscaler = StandardScaler()
datadf['x'] = sscaler.fit_transform(datadf[['x']])
datadf['y'] = sscaler.fit_transform(datadf[['y']])

def norm_distance(ori_distance, mu = mu, sd = sd):
	return (ori_distance - mu) / sd

def binary_distance(ori_distance, mu = mu):
	if ori_distance >= mu:
		return '1'
	else:
		return '0'

def get_img_array(path):
	cur = imread(path)
	cur = resize(cur,(height,width,3))
	return np.reshape(cur, (int(height*width*3)))

datadf['normalized_distance'] = datadf.distance.apply(norm_distance)
datadf['binary_distance'] = datadf.distance.apply(binary_distance)

os.chdir('/media/data_cifs/projects/prj_multi-cue/coordinates')

print('Save validated data frame')
datadf.to_parquet('data_1000_validated.parquet.gzip',compression='gzip')

print('loading img into array')
ftrs = ['x', 'y', 'binary_distance', 'normalized_distance', 'img_array']
n = int(datadf.shape[0] // 10)

for i in range(10):
	print('Processing the {} th file'.format(i))
	tem_df = datadf[i*n:(i+1)*n]
	tem_df['img_array'] = tem_df.path.progress_apply(get_img_array)
	tem_df.to_parquet('data_img_{}.parquet'.format(i))

new_df = datadf[ftrs]

os.chdir('/media/data_cifs/projects/prj_multi-cue/coordinates')
#print('Saving')
#new_df.to_parquet('data_with_img_all.parquet.gzip',compression='gzip')
#datadf.to_parquet('data_with_img.parquet.gzip',compression='gzip')
print('Success!')