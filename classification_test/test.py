import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
import sys
sys.path.append('../')
import math
import datetime
import traceback
import draw_graph
import write_gspread
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
from dataset4test import get_data
import dataset4test
from tqdm import tqdm
from wide_resnet import WideResNet

import matplotlib.pyplot as plt

def get_arguments():
    argp = ArgPar()
    hp = {"batch_size": 128,
           "lr": 1.0e-1,
           "momentum": 0.9,
           "weight_decay": 5.0e-4,
           "width_coef1": 10,
           "width_coef2": 10,
           "width_coef3": 10,
           "n_blocks1": 4,
           "n_blocks2": 4,
           "n_blocks3": 4,
           "drop_rates1": 0.3,
           "drop_rates2": 0.3,
           "drop_rates3": 0.3,
           "lr_decay": 0.2,
           "num_of_class": 10}

    for name, v in hp.items():
        argp.add_argument("-{}".format(name), type = type(v), default = v)
    
    parsed_parameters = argp.parse_args()
    HyperParameters = {}
    
    for k in hp.keys():
        HyperParameters[k] =  eval("parsed_parameters.{}".format(k))
        
    return HyperParameters

def accuracy4test(y, target):
    global pred_total, target_total, model
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    pred_total.extend(pred.cpu())
    target_total.extend(target.cpu())
    save_caption = "dataset;" + os.path.basename(dataset4test.dataset_folder) + " model;" + model 
    draw_graph.plot_confusion_matrix(target_total, pred_total, dataset4test.test_classes.keys(), save_caption=save_caption, save_place='./cross_test_results/')

    return acc

def print_result(values):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<20"
    f_float = "{}:<20.5f"

    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    print(f_vars.format(*values))


def test(device, optimizer, learner, test_data, loss_func):
    global target_total, pred_total, cm_cnt
    target_total = []
    pred_total = []
    cm_cnt = 0
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = learner(data)
        loss = loss_func(y, target)

        test_acc += accuracy4test(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()
    
    return float(test_acc) / n_test, test_loss / n_test

def main(learner):

    #now = datetime.datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == 'cuda':
        learner = torch.nn.DataParallel(learner, device_ids=[0, 1, 2]) # make parallel
    
    test_data = get_data(learner.batch_size) # get_data is only used here. the cause seems to be in test_data
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = learner.lr, \
                        momentum = learner.momentum, \
                        weight_decay = learner.weight_decay, \
                        nesterov = True \
                        )
    
    loss_func = nn.CrossEntropyLoss().cuda()


    rsl_keys = ["TestAcc", "TestLoss"]
    rsl = []
    y_out = 1.0e+8
    
    print_result(rsl_keys)
    global model
    model = "4frames-extracted_output_x_x_18_18_0_resized_32_32_0716_2155.pth" # set model here
    learner.load_state_dict(torch.load('../dataset/4frames-extracted_output_x_x_18_18_0_resized_32_32/log/0716_2155/' + model))  # set pretrained model here!
    learner.eval() # switch to test mode (make model not save the record of calculation)
        
    lr = optimizer.param_groups[0]["lr"]     

    with torch.no_grad():
        test_acc, test_loss = test(device, optimizer, learner, test_data, loss_func)

    time_now = str(datetime.datetime.today())
    rsl.append({k: v for k, v in zip(rsl_keys, [test_acc, test_loss])})
       
    y_out = min(y_out, test_loss)
    print_result(rsl[-1].values())

        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    print("Start Testing")
    print("")
    main(learner)

