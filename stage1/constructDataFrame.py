import os
import pandas as pd
import pickle
import pdb
from tqdm import tqdm
from tqdm import trange
import numpy as np

# build a dataframe with 'image_id', 'distance', 'path'
datadf = pd.DataFrame(columns=['image_id', 'distance', 'path'])

p = []
distance = []
image_id = []

print('Constructing DataFrame')
os.chdir('/media/data_cifs/projects/prj_multi-cue/zhujun_texture')

for i in range(1,101):
	# construc the df for images in one file
	file_idx = i

	# find label data from 'data_log.pickle'
	description_path = os.path.join('/media/data_cifs/projects/prj_multi-cue/zhujun_texture', str(file_idx), 'data_log.pickle')
	file = open(description_path,'rb')
	data = pickle.load(file)

	# def the function for image path
	def img_path_for_df(img_id, file_num = i, pattern='%04d.png', top='/media/data_cifs/projects/prj_multi-cue/zhujun_texture'):
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
	tem_dis = [i['diametric_distance'] for i in data.values()]
	for i in tem_dis:
		distance.append(i)
	tem_path = [img_path_for_df(i) for i in data.keys()]
	for i in tem_path:
		p.append(i)

datadf['image_id'] = image_id
datadf['distance'] = distance
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
datadf.to_parquet('datadf.parquet.gzip',compression='gzip')
