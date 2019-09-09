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
from dataset4regress_frame import get_data
import dataset4regress_frame
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
            if not os.path.exists(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i])):
                 os.makedirs(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i]))
            shutil.copyfile(paths_total[i], dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, 0, np.array(target_total_test)[i], np.array(int_y_total)[i]) + splited_path[-1])
            bar.update()
        bar.close()

    #draw_graph.yyplot_density(np.array(target_total_test), np.array(y_total_test), False, save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
    #draw_graph.plot_confusion_matrix(target_total_test, int_y_total, dataset4regress_frame.num_list, save_caption=dataset4regress_frame.dataset_folder, save_place=dataset4regress_frame.dataset_directory + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
    
    f = open(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + 'target_heatmap.txt', 'w')
    f2 = open(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + 'prediction_heatmap.txt', 'w')
    blank_target = np.zeros((dataset4regress_frame.ori_height, dataset4regress_frame.ori_width, 3), np.uint8)
    blank_target_pt = np.zeros((dataset4regress_frame.ori_height, dataset4regress_frame.ori_width, 3), np.uint8)
    blank_pred = np.zeros((dataset4regress_frame.ori_height, dataset4regress_frame.ori_width, 3), np.uint8)
    bar = tqdm(desc = "making heatmap...", total = len(np.array(target_total_test)), leave = False)
    
    for pt in range(len(dataset4regress_frame.df)):
        blank_target_pt[int(dataset4regress_frame.df.loc[pt, 'y']-16), int(dataset4regress_frame.df.loc[pt, 'x']-16)] = [18, 0, 230]
    target_total_pt = []
        
    for pt in range(len(np.array(target_total_test))):
        if pt!=0 and pt%dataset4regress_frame.ori_width==0:
            f.write("\n")
            f2.write("\n")
        img_x = pt%dataset4regress_frame.ori_width
        img_y = int(pt/dataset4regress_frame.ori_width)
        
        if target_total_test[pt]==0:
            blank_target[img_y, img_x] = [255,255,255]  # white
        elif target_total_test[pt]==1:
            blank_target[img_y, img_x] = [18, 0, 230]  # red
        elif target_total_test[pt]==2:
            blank_target[img_y, img_x] = [0, 152, 243]  # orange
        elif target_total_test[pt]==3:
            blank_target[img_y, img_x] = [0, 241, 255]  # yellow
        elif target_total_test[pt]==4:
            blank_target[img_y, img_x] = [31,195,143]  # light-green
        elif target_total_test[pt]==5:
            blank_target[img_y, img_x] = [68,153,0]  # green
        elif target_total_test[pt]==6:
            blank_target[img_y, img_x] = [150,158,0]  # blue-green
        elif target_total_test[pt]==7:
            blank_target[img_y, img_x] = [233,160,0]  # sky
        elif target_total_test[pt]==8:
            blank_target[img_y, img_x] = [183,104,0]  # blue
        elif target_total_test[pt]==9:
            blank_target[img_y, img_x] = [136,32,29]  # navy
        else:
            blank_target[img_y, img_x] = [255-target_total_test[pt]*15, 255-target_total_test[pt]*15, 255-target_total_test[pt]*15]
            
        if int_y_total[pt]==0:
            blank_pred[img_y, img_x] = [255,255,255]  # white     
        elif int_y_total[pt]==1:
            blank_pred[img_y, img_x] = [18, 0, 230]  # red
        elif int_y_total[pt]==2:
            blank_pred[img_y, img_x] = [0, 152, 243]  # orange
        elif int_y_total[pt]==3:
            blank_pred[img_y, img_x] = [0, 241, 255]  # yellow
        elif int_y_total[pt]==4:
            blank_pred[img_y, img_x] = [31,195,143]  # light-green
        elif int_y_total[pt]==5:
            blank_pred[img_y, img_x] = [68,153,0]  # green
        elif int_y_total[pt]==6:
            blank_pred[img_y, img_x] = [150,158,0]  # blue-green
        elif int_y_total[pt]==7:
            blank_pred[img_y, img_x] = [233,160,0]  # sky
        elif int_y_total[pt]==8:
            blank_pred[img_y, img_x] = [183,104,0]  # blue
        elif int_y_total[pt]==9:
            blank_pred[img_y, img_x] = [136,32,29]  # navy
        else:
            blank_pred[img_y, img_x] = [255-int_y_total[pt]*15, 255-int_y_total[pt]*15, 255-int_y_total[pt]*15]
            
        f.write("{} ".format(target_total_test[pt]))
        f2.write("{} ".format(int_y_total[pt]))
        
        if blank_target_pt[img_y, img_x,0]!=0 or blank_target_pt[img_y, img_x,1]!=0 or blank_target_pt[img_y, img_x,2]!=0:
            target_total_pt.append(1)
        else:
            target_total_pt.append(0)

        bar.update()
    bar.close()
    f.close()
    f2.close()
    
    #draw_graph.ppl_in_frame(np.array(target_total_test), dataset4regress_frame.ori_width, dataset4regress_frame.ori_height, 32,  'target_ppl_part_num', save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="ave")
    draw_graph.ppl_in_frame(32*32*np.array(target_total_pt), dataset4regress_frame.ori_width, dataset4regress_frame.ori_height, 32,  'target_ppl_part_num', save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="ave")
    draw_graph.ppl_in_frame(np.array(int_y_total), dataset4regress_frame.ori_width, dataset4regress_frame.ori_height, 32, 'pred_ppl_part_num', save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="ave")  ## change here according to CellDataset or FrameDataset.
    #draw_graph.ppl_in_frame(np.array(int_y_total)-np.array(target_total_test), dataset4regress_frame.ori_width, dataset4regress_frame.ori_height, 32, 'diff_ppl_part_num', save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="diff")
    draw_graph.ppl_in_frame(np.array(int_y_total)-32*32*np.array(target_total_pt), dataset4regress_frame.ori_width, dataset4regress_frame.ori_height, 32, 'diff_ppl_part_num', save_place=dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now), caltype="diff")
    
    cv2.imwrite(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + 'target_heatmap.jpg', blank_target)
    cv2.imwrite(dataset4regress_frame.dataset_directory + '/' + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + 'prediction_heatmap.jpg', blank_pred)  
    return test_loss / n_test

def main(learner):
    global now, model
    
    device = "cuda"
    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    test_data = get_data(learner.module.batch_size)
    
    now = datetime.datetime.now()
    if not os.path.exists(dataset4regress_frame.dataset_directory + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now)):
        os.makedirs(dataset4regress_frame.dataset_directory + dataset4regress_frame.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now))
    
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
    learner.load_state_dict(torch.load('../dataset/4frames-extracted_output_x_x_18_18_0_resized_32_32/log/0714_1948/' + model), strict=False)  # set pretrained model here!
    
    print("Start Testing")
    print("")
    main(learner)

