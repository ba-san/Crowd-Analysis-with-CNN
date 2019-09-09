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
files = "/dataset/loc-1_output_x_x_x_x_x_resized_x_x"

full_path = PWD + files
train_files = glob.glob(full_path + "/train/*")
test_files = glob.glob(full_path + "/test/*")
print(len(train_files))
print(len(test_files))


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
			if len(df_exact)>=1: 
				break

		print('{}:{}'.format(cnt, fname))
		cnt = cnt+1
		d['x'].append(float(df_exact['x'].values))
		d['y'].append(float(df_exact['y'].values))
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
			if len(df_exact)>=1: 
				break

		print('{}:{}'.format(cnt, fname))
		cnt = cnt+1
		d['x'].append(float(df_exact['x'].values))
		d['y'].append(float(df_exact['y'].values))
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
        img = Image.open(self.img_paths[index])

        if self.transform is not None:
            img = self.transform(img)

        location = self.y[index]
        path = self.img_paths[index]

        return img, location, path

    def __len__(self):
        return self.img_paths.shape[0]

    
    
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes
    dataset_directory = os.path.dirname(os.path.dirname(os.getcwd()))
    dataset_folder = files
   
    #normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616]) #this is for CIFAR100

    transform_train = transforms.Compose([
                                #transforms.Pad(4, padding_mode = 'reflect'), # https://stackoverflow.com/questions/52471817/performing-a-reflective-center-pad-on-an-image
                                #transforms.RandomCrop(32), # resized to 32x32
                                transforms.RandomHorizontalFlip(), # whether flip or not is random, not axis is random.
                                transforms.ToTensor()])
                                #normalize]) 
    transform_test = transforms.Compose([
                                transforms.ToTensor()])
                                #normalize])

                        
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
    
    return train_data, test_data
