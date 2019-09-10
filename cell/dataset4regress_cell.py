#https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb

import os
import cv2
import glob
import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

PWD = os.path.dirname(os.getcwd())
#files = "/dataset/4test-320x180_output"
files = "/dataset/4test-352x212_output"
fname_splited = os.path.basename(files).split('_')
full_path = PWD + files
dataset_directory = PWD
dataset_folder = files
ori_width = 352-32
ori_height = 212-32
culm_cnt = 0

num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # num of people in cropped imgs

csv_path=full_path + '/' + fname_splited[0] + '.csv'
df = pd.read_csv(csv_path, index_col=0)
frm_paths = df['image']
frm_name = frm_paths[1].split('/')[-1]
#reading_path = full_path + '/' + frm_name.split('_')[0] + '-full_' + frm_name.split('_')[1] + '_checked/LAST/0.jpg'
reading_path = full_path + '/' + frm_name + '_checked/LAST/0.jpg'
frame = cv2.imread(reading_path)
###################################################################

class CellDataset(Dataset):

    def __init__(self, transform=None):
        self.x = df['x'].values
        self.y = df['y'].values
        self.transform = transform

    def __getitem__(self, index):
        global frame_width, frame_height, frame, df, frm_paths, culm_cnt
        img_x = index%int((frame.shape[1]-32)/32)
        img_y = int(index/int((frame.shape[1]-32)/32))
        img = frame[16+32*img_y:16+min(32*(img_y+1), frame.shape[0])-1, 16+32*img_x:16+min(32*(img_x+1), frame.shape[1])-1]
        
        if self.transform is not None:
            img = self.transform(img)
            
        cnt = 0
        for pt in range(len(df)): # http://www.kisse-logs.com/2017/04/11/python-dataframe-drop/
            if df.loc[pt, 'image'] == frm_paths[1]: #double check
                pt_x = df.loc[pt, 'x']
                pt_y = df.loc[pt, 'y']
                if 16+32*img_x <= pt_x and pt_x <= 16+32*(img_x+1) and 16+32*img_y <= pt_y and pt_y <= 16+32*(img_y+1): # right left up down
                    cnt+=1
                    culm_cnt+=1

        label = cnt
        path = frm_paths[1] + '_checked/LAST/0.jpg'

        return img, label, path

    def __len__(self):
        global frame
        return int((frame.shape[0]-32)/32+.99999999999999)*int((frame.shape[1]-32)/32+.99999999999999)   
        
      
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes

    transform_test = transforms.Compose([
                                transforms.ToTensor()])

    test_dataset = CellDataset(
                        transform = transform_test
                        )
                        
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 64, pin_memory = True)
    
    return test_data
