import os
import glob
import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd

#https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path    
    
    
def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes
    dataset_directory = os.path.dirname(os.getcwd())
    dataset_folder = "./dataset/C0017_output_256_256_18_18_0_resized_32_32" #set dataset here
   
    #normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616]) #this is for CIFAR100

    transform_train = transforms.Compose([
                                #transforms.Pad(4, padding_mode = 'reflect'), # https://stackoverflow.com/questions/52471817/performing-a-reflective-center-pad-on-an-image
                                #transforms.RandomCrop(32), # resized to 32x32
                                transforms.RandomHorizontalFlip(), # whether flip or not is random, not axis is random.
                                transforms.ToTensor()])
                                #transforms.RandomErasing()])
                                #normalize]) 
    transform_test = transforms.Compose([
                                transforms.ToTensor()])
                                #normalize])

                        
    train_dataset = ImageFolderWithPaths(
                        root = dataset_directory  + dataset_folder[1:] + "/train",
                        transform = transform_train
                        )
    test_dataset = ImageFolderWithPaths(
                        root = dataset_directory  + dataset_folder[1:] + "/test",
                        transform = transform_test
                        )
                        
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 16, pin_memory = True)
    
    test_classes = test_dataset.class_to_idx
    print(test_dataset.class_to_idx)
    print('num of class:{}'.format(len(test_dataset.class_to_idx)))
    
    return train_data, test_data
