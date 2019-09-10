import csv
import os
import cv2
import numpy as np
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
import sys
sys.path.append('../')
import math
import shutil
import datetime
import traceback
import draw_graph
import write_gspread
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
from dataset4regress_cell import get_data
import dataset4regress_cell
from tqdm import tqdm
from wide_resnet_regress import WideResNet

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
           
def get_arguments():
    argp = ArgPar()
    hp = {"batch_size": 128,
          "lr": 1.0e-2,
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
          "lr_decay": 0.2}

    for name, v in hp.items():
        argp.add_argument("-{}".format(name), type = type(v), default = v)
    
    parsed_parameters = argp.parse_args()
    HyperParameters = {}
    
    for k in hp.keys():
        HyperParameters[k] =  eval("parsed_parameters.{}".format(k))
        
    return HyperParameters

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
    global target_total, pred_total
    target_total_test = []
    target_total_test_tensor = []
    y_total_test = []
    int_y_total = []
    paths_total = []
    test_loss, n_test = 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for batch_idx, (data, target, paths) in enumerate(test_data):
        data, target_tensor = data.to(device), target.to(device)

        y = learner(data)
        int_y = (y+0.5).int()
        
        y = y.view(1, -1)
        loss = loss_func(y, target_tensor.float())
        y = y.view(-1, 1)
        
        y_total_test.extend(y.cpu())
        target_total_test_tensor.extend(target_tensor.cpu())
        target_total_test.extend(target)
        int_y_total.extend(int_y.cpu())
        paths_total.extend(paths)
        
        test_loss += loss.item() * target_tensor.size(0)
        n_test += target_tensor.size(0)
        
        bar.set_description("Loss(MSE): {0:.6f}".format(test_loss / n_test))

        bar.update()
    bar.close()
    
    if False:
        bar = tqdm(desc = "False-pred...", total = len(target_total_test), leave = False)
        for i in range(len(target_total_test)):
            splited_path = paths_total[i].split('/')
            if not os.path.exists(dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i])):
                 os.makedirs(dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i]))
            shutil.copyfile(paths_total[i], dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i]) + splited_path[-1])
            bar.update()
        bar.close()
   
    draw_graph.ppl_in_cell(np.array(target_total_test), dataset4regress_cell.ori_width, dataset4regress_cell.ori_height, 32,  'target_ppl_part_num', save_place=dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="sum")
    draw_graph.ppl_in_cell(np.array(int_y_total), dataset4regress_cell.ori_width, dataset4regress_cell.ori_height, 32, 'pred_ppl_part_num', save_place=dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="sum")  ## change here according to CellDataset or FrameDataset.
    draw_graph.ppl_in_cell(np.array(int_y_total)-np.array(target_total_test), dataset4regress_cell.ori_width, dataset4regress_cell.ori_height, 32, 'diff_ppl_part_num', save_place=dataset4regress_cell.dataset_directory + '/' + dataset4regress_cell.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="diff")
    
    return test_loss / n_test

def main(learner):
    global now, model
    
    device = "cuda"
    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    test_data = get_data(learner.module.batch_size)
    
    now = datetime.datetime.now()
    if not os.path.exists(dataset4regress_cell.dataset_directory + dataset4regress_cell.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now)):
        os.makedirs(dataset4regress_cell.dataset_directory + dataset4regress_cell.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now))
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = learner.module.lr, \
                        momentum = learner.module.momentum, \
                        weight_decay = learner.module.weight_decay, \
                        nesterov = True \
                        )
    
    loss_mse = nn.MSELoss().cuda()

    rsl_keys = ["lr", "epoch", "TestLoss", "Time"]
    rsl = []
    
    print_result(rsl_keys)
    
    learner.eval() # switch to test mode (make model not save the record of calculation)
        
    lr = optimizer.param_groups[0]["lr"]

    with torch.no_grad():
        test_loss = test(device, optimizer, learner, test_data, loss_mse)
        s = 'Test Loss: %.2f' % (test_loss)
        print(s)
            
    time_now = str(datetime.datetime.today())
    rsl.append({k: v for k, v in zip(rsl_keys, [lr, test_loss, time_now])})
    print_result(rsl[-1].values())     

        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    model = "4frames-extracted_output_x_x_18_18_0_resized_32_32_0714_1948.pth" #set model here
    learner.load_state_dict(torch.load('../dataset/resized/4frames-extracted_output_x_x_18_18_0_resized_32_32/log/0714_1948/' + model), strict=False)  # set pretrained model here!
    
    print("Start Testing")
    print("")
    main(learner)
