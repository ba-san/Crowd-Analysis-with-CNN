import os
import torch
import random
from torchvision import datasets
from torchvision import transforms
import numpy as np

def get_data(batch_size):
    global dataset_folder, dataset_directory, test_classes
    dataset_directory = os.path.dirname(os.getcwd())
    dataset_folder = "./dataset/C0017_output_256_256_18_18_0_resized_32_32" #set dataset here
   
    normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                          std=[0.2471, 0.2435, 0.2616])

    transform_test = transforms.Compose([
                                transforms.ToTensor(), 
                                normalize])
                                            
    test_dataset = datasets.ImageFolder(
                        root = dataset_directory  + dataset_folder[1:] + "/test",
                        transform = transform_test
                        )
                                        
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    
    test_classes = test_dataset.class_to_idx
    print(test_dataset.class_to_idx)

    return test_data
