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

    transform_train = transforms.Compose([
                                transforms.RandomHorizontalFlip(), # whether flip or not is random, not axis is random.
                                transforms.ToTensor()])
    transform_test = transforms.Compose([
                                transforms.ToTensor()])
                        
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
