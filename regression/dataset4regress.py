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

PWD = os.path.dirname(os.getcwd())
files = "/dataset/C0017_output_256_256_18_18_0_resized_32_32" #set dataset here
full_path = PWD + files
train_files = glob.glob(full_path + "/train/*/*")
test_files = glob.glob(full_path + "/test/*/*")
print(len(train_files))
print(len(test_files))

# train
d = {}

d['num'] = []
d['file'] = []
d['path'] = []

for f in train_files:
    f_split = f.split('/')
    num = f_split[-2]
    fname = f_split[-1]
        
    d['num'].append(num)
    d['file'].append(fname)
    d['path'].append(f)
    
df_train = pd.DataFrame.from_dict(d)
df_train.to_csv(full_path + '/df_train.csv')

# test
d = {}

d['num'] = []
d['file'] = []
d['path'] = []

for f in test_files:
    f_split = f.split('/')
    num = f_split[-2]
    fname = f_split[-1]
        
    d['num'].append(num)
    d['file'].append(fname)
    d['path'].append(f)
    
df_test = pd.DataFrame.from_dict(d)
df_test.to_csv(full_path + '/df_test.csv')

num_ppl = np.unique(df_train['num'].values).shape[0]
num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
print('num of class:{}'.format(num_ppl))

###################################################################

NUM_CLASSES = num_ppl

class CrowdDataset(Dataset):
    """Custom Dataset for loading Crowd images"""

    def __init__(self, csv_path, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.y = df['num'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        path = self.img_paths[index]

        return img, label, path

    def __len__(self):
        return self.y.shape[0]
    
    
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes
    dataset_directory = os.path.dirname(os.getcwd())
    dataset_folder = files

    transform_train = transforms.Compose([
                                transforms.RandomHorizontalFlip(), # whether flip or not is random, not axis is random.
                                transforms.ToTensor()])
    transform_test = transforms.Compose([
                                transforms.ToTensor()])
            
    train_dataset = CrowdDataset(
                        csv_path=full_path + '/df_train.csv',
                        transform = transform_train
                        )
    test_dataset = CrowdDataset(
                        csv_path=full_path + '/df_test.csv',
                        transform = transform_test
                        )
                        
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    
    return train_data, test_data
