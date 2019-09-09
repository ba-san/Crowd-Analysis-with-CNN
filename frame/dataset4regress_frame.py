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
files = "/dataset/4test-320x180_output"
fname_splited = os.path.basename(files).split('_')
full_path = PWD + files
dataset_directory = PWD
dataset_folder = files
ori_width = 320
ori_height = 180

num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

csv_path=full_path + '/' + fname_splited[0] + '.csv'
df = pd.read_csv(csv_path, index_col=0)
frm_paths = df['image']
frame = cv2.imread(frm_paths[1] + '_checked/LAST/0.jpg')
###################################################################

class FrameDataset(Dataset):

    def __init__(self, transform=None):
        self.x = df['x'].values
        self.y = df['y'].values
        self.transform = transform

    def __getitem__(self, index):
        global frame_width, frame_height, frame, df, frm_paths
        img_x = index%(frame.shape[1]-32)
        frame_width = frame.shape[1]-32
        frame_height = frame.shape[0]-32
        img_y = int(index/(frame.shape[1]-32))
        img = frame[img_y:img_y+32, img_x:img_x+32]

        if self.transform is not None:
            img = self.transform(img)
            
        cnt = 0

        for pt in range(len(df)): # http://www.kisse-logs.com/2017/04/11/python-dataframe-drop/
            if df.loc[pt, 'image'] == frm_paths[1]: #double check
                pt_x = df.loc[pt, 'x']
                pt_y = df.loc[pt, 'y']
                if img_x <= pt_x and pt_x <= img_x + 32 and img_y <= pt_y and pt_y <= img_y + 32: # right left up down
                    cnt+=1

        label = cnt
        path = frm_paths[1] + '_checked/LAST/0.jpg'

        return img, label, path

    def __len__(self):
        global frame
        return (frame.shape[0]-32)*(frame.shape[1]-32)
        
      
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes


    transform_test = transforms.Compose([
                                transforms.ToTensor()])

    test_dataset = FrameDataset(
                        transform = transform_test
                        )
                        
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 64, pin_memory = True)
    
    return test_data
