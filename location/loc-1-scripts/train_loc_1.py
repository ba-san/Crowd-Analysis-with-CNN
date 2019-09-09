import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
import sys
sys.path.append('../../')
import cv2
import math
import glob
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
from dataset4loc_1 import get_data
import dataset4loc_1
from tqdm import tqdm
from wide_resnet_loc_1 import WideResNet

from joblib import Parallel, delayed
import matplotlib.pyplot as plt


####### original parameters #######

    #hp = {"batch_size": 128,
           #"lr": 1.0e-1,
           #"momentum": 0.9,
           #"weight_decay": 5.0e-4,
           #"width_coef1": 10,
           #"width_coef2": 10,
           #"width_coef3": 10,
           #"n_blocks1": 4,
           #"n_blocks2": 4,
           #"n_blocks3": 4,
           #"drop_rates1": 0.3,
           #"drop_rates2": 0.3,
           #"drop_rates3": 0.3,
           #"lr_decay": 0.2}
           
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

def train(device, optimizer, learner, train_data, loss_func):
    target_total_train = []
    loc_total_train = []
    train_loss, n_train = 0, 0
    lr = optimizer.param_groups[0]["lr"]
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for batch_idx, (data, target, paths) in enumerate(train_data):
        data = data.to(device)
        
        loc = learner(data)

        #loc = loc.view(1, -1) 
        loss_x = loss_func(loc[:, 0], target[0].to(device).float())
        loss_y = loss_func(loc[:, 1], target[1].to(device).float())
        #Euclidean loss (without sqrt)
        loss = loss_x + loss_y
        #loc = loc.view(-1, 1)

        loc_total_train.extend(loc)
        target_total_train.extend(target)

        optimizer.zero_grad() # clears the gradients of all optimized tensors for next train.
        loss.backward()  # backpropagation, compute gradients
        optimizer.step() # apply gradients. renew learning rate.

        train_loss += loss.item() * len(target[0]) # loss.item() is a loss num, but not a tensor. target.size(0) is batch size.
        n_train += len(target[0])

        bar.set_description("Loss(MSE): {0:.6f}".format(train_loss / n_train))
        
        bar.update()
    bar.close()

    return train_loss / n_train

def test(device, optimizer, learner, test_data, loss_func):
    global target_total, pred_total, epoch
    loc_check = True
    target_total_test = []
    target_total_test_tensor = []
    loc_total_test = []
    paths_total = []
    test_loss, n_test = 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for batch_idx, (data, target, paths) in enumerate(test_data):
        data = data.to(device)
        
        loc = learner(data)
        
        #y = y.view(1, -1)
        loss_x = loss_func(loc[:, 0], target[0].to(device).float())
        loss_y = loss_func(loc[:, 1], target[1].to(device).float())
        #Euclidian loss (without sqrt)
        loss = loss_x + loss_y
        #y = y.view(-1, 1)

        loc_total_test.extend(loc)
        target_total_test.extend(target)
        paths_total.extend(paths)
        
        test_loss += loss.item() * len(target[0])
        n_test += len(target[0])
        
        bar.set_description("Loss(MSE): {0:.6f}".format(test_loss / n_test))

        bar.update()
    bar.close()
    
    
    #if loc_check==True:
    if (epoch>1 and epoch%40==0) or epoch==199:
        hp_for_record= get_arguments()
        bs = hp_for_record["batch_size"]
        if os.path.exists(dataset4loc_1.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch)):
            shutil.rmtree(dataset4loc_1.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch))
        shutil.copytree(os.path.join(dataset4loc_1.full_path, "test"), dataset4loc_1.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch))
        
        bar = tqdm(desc = "dotting", total = len(paths_total), leave = False)
        f = open(dataset4loc_1.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch) + str(epoch) + '-pred-locations.txt', 'w')
        for i in range(len(paths_total)):
            path_separated = paths_total[i].split('/')
            img_lists = glob.glob(dataset4loc_1.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/*/*'.format(now, epoch))
            img_file = [s for s in img_lists if path_separated[-1] in s]
            img = cv2.imread(img_file[0])
            
            f.write('{}\n'.format(os.path.basename(img_file[0])))
            f.write('prediction  :{},{}\n'.format(int(loc_total_test[i][0]), int(loc_total_test[i][1])))
            f.write('ground truth:{},{}\n\n'.format(int(target_total_test[int(i/bs)][i%bs]), int(target_total_test[int(i/bs)][i%bs])))
               
            if 0<=int(loc_total_test[i][1])<img.shape[1] and 0<=int(loc_total_test[i][0])<img.shape[0]:
                img[int(loc_total_test[i][1]), int(loc_total_test[i][0])] = [0, 0, 255]
                if int(loc_total_test[i][1])+1 < img.shape[1]:
                    img[int(loc_total_test[i][1])+1, int(loc_total_test[i][0])] = [0, 0, 255]
                if int(loc_total_test[i][1])-1 >= 0:
                    img[int(loc_total_test[i][1])-1, int(loc_total_test[i][0])] = [0, 0, 255]
                if int(loc_total_test[i][0])+1 < img.shape[0]:
                    img[int(loc_total_test[i][1]), int(loc_total_test[i][0])+1] = [0, 0, 255]
                if int(loc_total_test[i][0])-1 >= 0:    
                    img[int(loc_total_test[i][1]), int(loc_total_test[i][0])-1] = [0, 0, 255]
            cv2.imwrite(img_file[0], img)
            bar.update()
        bar.close()
        f.close()

    return test_loss / n_test


global now
def main(learner):

    device = "cuda"
    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    train_data, test_data = get_data(learner.module.batch_size)
    
    global now
    now = datetime.datetime.now()
    if not os.path.exists(dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now)):
        os.makedirs(dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}'.format(now, now))
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        #lr = learner.lr, \
                        lr = learner.module.lr, \
                        #momentum = learner.momentum, \
                        momentum = learner.module.momentum, \
                        #weight_decay = learner.weight_decay, \
                        weight_decay = learner.module.weight_decay, \
                        nesterov = True \
                        )
    
    loss_mse = nn.MSELoss().cuda()

    #milestones = learner.lr_step
    milestones = learner.module.lr_step
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.lr_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.module.lr_decay)

    rsl_keys = ["lr", "epoch", "TrainLoss", "TestLoss", "Time"]
    rsl = []
    
    print_result(rsl_keys)
    
    global patience, counter, best_loss, epoch
    patience = 200 #set any adequate number here
    counter = 0
    earlystopper = 0
    best_loss = None
    
    #for epoch in range(learner.epochs):
    for epoch in range(learner.module.epochs):
        
        lr = optimizer.param_groups[0]["lr"] 
        train_loss = train(device, optimizer, learner, train_data, loss_mse) 
        
        learner.eval() # switch to test mode (make model not save the record of calculation)

        with torch.no_grad():
            test_loss = test(device, optimizer, learner, test_data, loss_mse)
            s = 'Test Loss: %.2f' % (test_loss)
            print(s)
            
            ### early stopping ###
            if best_loss is None:
                best_loss = test_loss
            elif test_loss > best_loss:
                counter += 1
                print("EarlyStopping: %i / %i" % (counter, patience))
                if counter >= patience:
                    print("EarlyStopping: Stop training")
                    earlystopper = 1
            else:
                best_loss = test_loss
                counter = 0
            ######################


        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_loss, test_loss, time_now])})
     
        #draw_graph.draw_graph_regress(learner.epochs, epoch, train_loss, test_loss, os.path.basename(dataset4loc_1.dataset_folder), save_place=dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
        draw_graph.draw_graph_regress(learner.module.epochs, epoch, train_loss, test_loss, os.path.basename(dataset4loc_1.dataset_folder), save_place=dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
        
        hp_for_record= get_arguments()
        otherparams = []
        otherparams.append(hp_for_record["batch_size"])
        otherparams.append(hp_for_record["lr"])
        otherparams.append(hp_for_record["momentum"])
        otherparams.append(hp_for_record["weight_decay"])
        otherparams.append(hp_for_record["width_coef1"])
        otherparams.append(hp_for_record["width_coef2"])
        otherparams.append(hp_for_record["width_coef3"])
        otherparams.append(hp_for_record["n_blocks1"])
        otherparams.append(hp_for_record["n_blocks2"])
        otherparams.append(hp_for_record["n_blocks3"])
        otherparams.append(hp_for_record["drop_rates1"])
        otherparams.append(hp_for_record["drop_rates2"])
        otherparams.append(hp_for_record["drop_rates3"])
        otherparams.append(hp_for_record["lr_decay"])
        otherparams.append(time_now)

        save_place = dataset4loc_1.dataset_directory + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now)
        write_gspread.update_gspread(dataset4loc_1.dataset_folder, 'WRN', dataset4loc_1.dataset_directory, now, 'N/A(regress)', train_loss, 'N/A(regress)', test_loss, epoch+1, learner.module.epochs, False, save_place, otherparams)
        
        print_result(rsl[-1].values())
        scheduler.step()
        
        #torch.save(learner.state_dict(), dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + os.path.basename(dataset4loc_1.dataset_folder) + '_{0:%m%d}_{0:%H%M}.pth'.format(now, now))
        torch.save(learner.module.state_dict(), dataset4loc_1.dataset_directory  + dataset4loc_1.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + os.path.basename(dataset4loc_1.dataset_folder) + '_{0:%m%d}_{0:%H%M}.pth'.format(now, now))
        
        if earlystopper == 1:
            write_gspread.update_gspread(dataset4loc_1.dataset_folder, 'WRN', dataset4loc_1.dataset_directory, now, 'N/A(regress)', train_loss, 'N/A(regress)', test_loss, epoch+1, learner.module.epochs, True, save_place, otherparams)
            break

        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    print("Start Training")
    print("")
    main(learner)

