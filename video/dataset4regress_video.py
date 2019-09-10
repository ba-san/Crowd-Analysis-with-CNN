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
files = "/dataset/C0003-heatmap2_320x180/"
fname_splited = os.path.basename(files).split('_')
full_path = PWD + files
dataset_directory = PWD
dataset_folder = files

xSize = 320
ySize = 180
timeDepth = 3
channels = 3

num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

def readVideo(videoFile):
	global xSize, ySize, timeDepth, channels
	cap = cv2.VideoCapture(videoFile)
	frames = torch.FloatTensor(timeDepth, ySize, xSize, channels)
	failedClip = False
	for f in range(timeDepth):

		ret, frame = cap.read()
		if ret:
			frame = torch.from_numpy(frame)
			frames[f, :, :, :] = frame

		else:
			print("Skipped!")
			failedClip = True
			break

	return frames, failedClip
        
        
global clip, failedClip
clip, failedClip = readVideo(dataset_directory + dataset_folder + 'C0003-heatmap2_320x180.mp4')

clip = np.array(clip)
blank_clip = np.zeros((timeDepth, ySize+32, xSize+32, 3), np.uint8)

if not os.path.exists(dataset_directory + dataset_folder + '/video2img/'):
    os.makedirs(dataset_directory + dataset_folder + '/video2img/')
    
for i in range(timeDepth):
    blank_clip[i, 16:ySize+16,16:xSize+16, :] = clip[i]
    cv2.imwrite(dataset_directory + dataset_folder + '/video2img/{}_ori.jpg'.format(i), blank_clip[i])

clip = blank_clip
  
###################################################################

# https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset/blob/master/videoDataset.py
class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, mean=None, transform=None):
        """
		Args:
			fideoFile (string): path of the video.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
			mean: Mean valuse of the training set videos over each channel
		"""

        self.mean = mean
        self.transform = transform

    def __len__(self):
        global xSize, ySize, timeDepth
        return timeDepth*xSize*ySize

    def __getitem__(self, index):
        global xSize, ySize, clip, failedClip
        frm_num = int(index/(xSize*ySize))
        index = index%(xSize*ySize)
        
        img_x = index%xSize
        img_y = int(index/xSize)
        
        img = clip[frm_num][img_y:img_y+32, img_x:img_x+32] 
        img = self.transform(img)
        
        return img
      
def get_data(batch_size):
    transform_test = transforms.Compose([
                                transforms.ToTensor()])

    test_dataset = videoDataset(
                        mean = None,
                        transform = transform_test
                        )
                        
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 128, pin_memory = True)
    
    return test_data
