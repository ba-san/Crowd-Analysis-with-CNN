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
from dataset4regress_video import get_data
import dataset4regress_video
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
    
    for batch_idx, (data) in enumerate(test_data):
        data = data.to(device)

        y = learner(data)
        int_y = (y+0.5).int()

        y_total_test.extend(y.cpu())
        int_y_total.extend(int_y.cpu())

        bar.update()
    bar.close()
    
    blank_pred = np.zeros((dataset4regress_video.timeDepth, dataset4regress_video.ySize, dataset4regress_video.xSize, 3), np.uint8)
    bar = tqdm(desc = "making heatmap...", total = len(np.array(int_y_total)), leave = False)
    for pt in range(len(np.array(int_y_total))):
        frm_num = int(pt/(dataset4regress_video.xSize*dataset4regress_video.ySize))
        pt_new = pt%(dataset4regress_video.xSize*dataset4regress_video.ySize)
        img_x = pt_new%dataset4regress_video.xSize
        img_y = int(pt_new/dataset4regress_video.xSize)
            
        if int_y_total[pt]==0:   
            blank_pred[frm_num, img_y, img_x, :] = [255,255,255]  # white     
        elif int_y_total[pt]==1:
            blank_pred[frm_num, img_y, img_x, :] = [18, 0, 230]  # red
        elif int_y_total[pt]==2:
            blank_pred[frm_num, img_y, img_x, :] = [0, 152, 243]  # orange
        elif int_y_total[pt]==3:
            blank_pred[frm_num, img_y, img_x, :] = [0, 241, 255]  # yellow
        elif int_y_total[pt]==4:
            blank_pred[frm_num, img_y, img_x, :] = [31,195,143]  # light-green
        elif int_y_total[pt]==5:
            blank_pred[frm_num, img_y, img_x, :] = [68,153,0]  # green
        elif int_y_total[pt]==6:
            blank_pred[frm_num, img_y, img_x, :] = [150,158,0]  # blue-green
        elif int_y_total[pt]==7:
            blank_pred[frm_num, img_y, img_x, :] = [233,160,0]  # sky
        elif int_y_total[pt]==8:
            blank_pred[frm_num, img_y, img_x, :] = [183,104,0]  # blue
        elif int_y_total[pt]==9:
            blank_pred[frm_num, img_y, img_x, :] = [136,32,29]  # navy
        else:
            blank_pred[frm_num, img_y, img_x, :] = [255-int_y_total[pt]*15, 255-int_y_total[pt]*15, 255-int_y_total[pt]*15]

        bar.update()
    bar.close()
    
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/heatmap-video.mp4'.format(now), fourcc, 30, (dataset4regress_video.xSize, dataset4regress_video.ySize))
    video2 = cv2.VideoWriter(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/pred-ppl-part-num-video.mp4'.format(now), fourcc, 30, (640, 480))
    for i in range(dataset4regress_video.timeDepth):
        video.write(blank_pred[i])
        draw_graph.ppl_in_frame(np.array(int_y_total[i*dataset4regress_video.xSize*dataset4regress_video.ySize:(i+1)*dataset4regress_video.xSize*dataset4regress_video.ySize-1]), dataset4regress_video.xSize, dataset4regress_video.ySize, 32, 'pred_ppl_part_num;{}'.format(i), save_place=dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now), caltype="ave")
        area_map = cv2.imread(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/pred_ppl_part_num;{1}.png'.format(now, i))
        video2.write(area_map)
        cv2.imwrite(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/{1}.jpg'.format(now, i), blank_pred[i])
    video.release()
    video2.release()


def main(learner):
    global now, model
    
    device = "cuda"
    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    test_data = get_data(learner.module.batch_size)
    
    now = datetime.datetime.now()
    if not os.path.exists(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now)):
        os.makedirs(dataset4regress_video.dataset_directory + dataset4regress_video.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now))
    
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
    
    learner.eval() # switch to test mode (make model not save the record of calculation)
        
    lr = optimizer.param_groups[0]["lr"]

    with torch.no_grad():
        test(device, optimizer, learner, test_data, loss_mse)   

        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    model = "4frames-extracted_output_x_x_18_18_0_resized_32_32_0714_1948.pth" # set model here
    learner.load_state_dict(torch.load('../dataset/4frames-extracted_output_x_x_18_18_0_resized_32_32/log/0714_1948/' + model), strict=False)  # set pretrained model here!
    
    print("Start Testing")
    print("")
    main(learner)
