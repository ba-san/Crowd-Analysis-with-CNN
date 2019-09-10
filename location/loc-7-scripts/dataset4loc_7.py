#https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb

import os
import glob
import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

PWD = os.path.dirname(os.path.dirname(os.getcwd()))
files = "/dataset/resized/loc-7/loc-7-extensive-extracted-raster_output_x_x_x_x_x_resized_x_x"

full_path = PWD + files
train_files = glob.glob(full_path + "/train/7/*")
test_files = glob.glob(full_path + "/test/7/*")
print(len(train_files))
print(len(test_files))

num_loc = 7  # must be more than 1

if os.path.exists(full_path + "/df_train.csv"):
	pass
else:
	d = {}
	cnt = 0

	d['x'] = []
	d['y'] = []
	d['file'] = []
	d['path'] = []

	for f in train_files:
		f_split = f.split('/')
		num = f_split[-2]
		fname = f_split[-1]
		fname4exact = fname[:-12] + '.jpg'
		print(fname4exact)
		df_exact = pd.DataFrame(columns=['image', 'x', 'y', 'color', 'outer_circle'])
    
		any_csv = glob.glob(full_path + "/csv/train/*.csv")
		for csv in any_csv:
			df = pd.read_csv(csv)
			df_exact = df[df['image'].str.contains(fname4exact)]
			if len(df_exact)>=num_loc:
				break

		print('{}:{}'.format(cnt, fname))
		cnt = cnt+1
		for i in range(num_loc):
			d['x'].append(float(df_exact['x'].values[i]))
			d['y'].append(float(df_exact['y'].values[i]))
			d['file'].append(fname)
			d['path'].append(f)
    
	df_train = pd.DataFrame.from_dict(d)
	df_train.to_csv(full_path + '/df_train.csv')


	d = {}
	cnt = 0

	d['x'] = []
	d['y'] = []
	d['file'] = []
	d['path'] = []

	for f in test_files:
		f_split = f.split('/')
		num = f_split[-2]
		fname = f_split[-1]
		fname4exact = fname[:-12] + '.jpg'
		df_exact = pd.DataFrame(columns=['image', 'x', 'y', 'color', 'outer_circle'])
    
		any_csv = glob.glob(full_path + "/csv/test/*.csv")
		for csv in any_csv:
			df = pd.read_csv(csv)
			df_exact = df[df['image'].str.contains(fname4exact)]
			if len(df_exact)>=num_loc:
				break

		print('{}:{}'.format(cnt, fname))
		cnt = cnt+1
		for i in range(num_loc):
			d['x'].append(float(df_exact['x'].values[i]))
			d['y'].append(float(df_exact['y'].values[i]))
			d['file'].append(fname)
			d['path'].append(f)
    
	df_test = pd.DataFrame.from_dict(d)
	df_test.to_csv(full_path + '/df_test.csv')
	## END OF ELSE ##


###################################################################

class LocationDataset(Dataset):
    """Custom Dataset for loading Crowd images"""

    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.y = [[a, b] for (a, b) in zip(df['x'].values, df['y'].values)]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index*7])

        if self.transform is not None:
            img = self.transform(img)

        location = [self.y[index*7], self.y[index*7+1], self.y[index*7+2], self.y[index*7+3], self.y[index*7+4], self.y[index*7+5], self.y[index*7+6]]
        path = self.img_paths[index*7]

        return img, location, path

    def __len__(self):
        return int(self.img_paths.shape[0]/7)
    
    
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes
    dataset_directory = os.path.dirname(os.path.dirname(os.getcwd()))
    dataset_folder = files
    list4flip = []
   
    #normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616]) #this is for CIFAR100
    normalize = transforms.Normalize(mean=[9.516211, 6.992552, 1.6308211], std=[0.23743322, 0.23230238, 0.24629381])

    transform_train = transforms.Compose([
                                #transforms.RandomHorizontalFlipCustom(list4flip),
                                transforms.ToTensor(),
                                normalize])
    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                normalize])

                        
    train_dataset = LocationDataset(
                        csv_path=full_path + '/df_train.csv',
                        transform = transform_train
                        )
    test_dataset = LocationDataset(
                        csv_path=full_path + '/df_test.csv',
                        transform = transform_test
                        )
                        
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    
    #### https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    ######################   for normalization   ######################
    #######!!   CHANGE BATCH SIZE BEFORE CHECKING STATISTICS   !!######
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, (data, target, paths)in enumerate(train_data, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data.numpy()
    
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
    
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    print('pop_mean:{}, pop_std0:{}'.format(pop_mean, pop_std0))
    #change batchsize also!!!  #exit()
    ###################################################################

    return train_data, test_data
