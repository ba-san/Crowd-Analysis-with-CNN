import csv
import os
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
from dataset_NOCHANGE import get_data
import dataset_NOCHANGE
from tqdm import tqdm
from wide_resnet_NOCHANGE import WideResNet

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
           #"lr_decay": 0.2,
           #"num_of_class": 10}
           
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
           "lr_decay": 0.2,
           "num_of_class": 10}

    for name, v in hp.items():
        argp.add_argument("-{}".format(name), type = type(v), default = v)
    
    parsed_parameters = argp.parse_args()
    HyperParameters = {}
    
    for k in hp.keys():
        HyperParameters[k] =  eval("parsed_parameters.{}".format(k))
        
    return HyperParameters

def accuracy(y, target):
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()

    return acc

global pred_total, target_total
def accuracy4test(y, target_tensor, target, paths):
    global pred_total, target_total, now, epoch, counter, patience, all_epoch
    
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target_tensor.data.view_as(pred)).cpu().sum()
    pred_total.extend(pred.cpu())
    target_total.extend(target_tensor.cpu())
    
    if counter+1 >= patience or epoch == all_epoch-1: # or epoch%30==0: 
        for i in range(len(target_tensor.cpu())):
            if not os.path.exists(dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, epoch, target[i], np.array(pred.cpu()[i]))):
                os.makedirs(dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, epoch, target[i], np.array(pred.cpu()[i])))
            splited_path = paths[i].split('/')
            shutil.copyfile(paths[i], dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/{2}/{3}/'.format(now, epoch, target[i], np.array(pred.cpu()[i])) + splited_path[-1])
        draw_graph.plot_confusion_matrix(target_total, pred_total, dataset_NOCHANGE.test_classes.keys(), save_caption=dataset_NOCHANGE.dataset_folder, save_place=dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/false_pred_{1}/'.format(now, epoch))

    
    return acc, pred_total, target_total

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
    train_acc, train_loss, n_train = 0, 0, 0
    lr = optimizer.param_groups[0]["lr"]
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for data, target, paths in train_data:
        data, target = data.to(device), target.to(device)
        
        y = learner(data)
        loss = loss_func(y, target)
        optimizer.zero_grad() # clears the gradients of all optimized tensors.
        loss.backward() 
        optimizer.step() # renew learning rate

        train_acc += accuracy(y, target)

        train_loss += loss.item() * target.size(0) # loss.item() is a loss num, but not a tensor. target.size(0) is batch size.
        n_train += target.size(0)

        bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
        previous_loss = train_loss / n_train
        bar.update()
    bar.close()
    
    return float(train_acc) / n_train, train_loss / n_train

def test(device, optimizer, learner, test_data, loss_func):
    global target_total, pred_total
    target_total = []
    pred_total = []
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for data, target, paths in test_data:
        data, target_tensor = data.to(device), target.to(device)
        	
        y = learner(data)
        loss = loss_func(y, target_tensor)
        
        acc, pred_total, target_total = accuracy4test(y, target_tensor, target, paths)
        test_acc += acc
        test_loss += loss.item() * target_tensor.size(0)
        n_test += target_tensor.size(0)

        bar.update()
    bar.close()
    
    draw_graph.plot_confusion_matrix(target_total, pred_total, dataset_NOCHANGE.test_classes.keys(), save_caption=dataset_NOCHANGE.dataset_folder, save_place=dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))


    return float(test_acc) / n_test, test_loss / n_test

global now
def main(learner):

    device = "cuda"
    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    train_data, test_data = get_data(learner.module.batch_size)
    
    global now
    now = datetime.datetime.now()
    if not os.path.exists(dataset_NOCHANGE.dataset_directory + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}'.format(now, now)):
        os.makedirs(dataset_NOCHANGE.dataset_directory + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}'.format(now, now))
    
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
    
    loss_func = nn.CrossEntropyLoss().cuda()

    #milestones = learner.lr_step
    milestones = learner.module.lr_step
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.lr_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.module.lr_decay)


    rsl_keys = ["lr", "epoch", "TrainAcc", "TrainLoss", "TestAcc", "TestLoss", "Time"]
    rsl = []
    
    print_result(rsl_keys)
    
    global patience, counter, best_loss, epoch, all_epoch
    patience = 20 #set any adequate number here
    counter = 0
    earlystopper = 0
    best_loss = None
    all_epoch = learner.module.epochs
    
    for epoch in range(all_epoch):
        
        lr = optimizer.param_groups[0]["lr"] 
        train_acc, train_loss = train(device, optimizer, learner, train_data, loss_func) 
        
        learner.eval() # switch to test mode (make model not save the record of calculation)

        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, learner, test_data, loss_func)
            
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
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
     
        #draw_graph.draw_graph(all_epoch, epoch, train_acc, train_loss, test_acc, test_loss, os.path.basename(dataset_NOCHANGE.dataset_folder), save_place=dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
        draw_graph.draw_graph(all_epoch, epoch, train_acc, train_loss, test_acc, test_loss, os.path.basename(dataset_NOCHANGE.dataset_folder), save_place=dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now))
        
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

        save_place = dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now)
        write_gspread.update_gspread(dataset_NOCHANGE.dataset_folder, 'WRN', dataset_NOCHANGE.dataset_directory, now, train_acc, train_loss, test_acc, test_loss, epoch+1, learner.module.epochs, False, save_place, otherparams)
        
        print_result(rsl[-1].values())
        scheduler.step()
        
        #torch.save(learner.state_dict(), dataset_NOCHANGE.dataset_directory  + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + os.path.basename(dataset_NOCHANGE.dataset_folder) + '_{0:%m%d}_{0:%H%M}.pth'.format(now, now))
        torch.save(learner.module.state_dict(), dataset_NOCHANGE.dataset_directory + dataset_NOCHANGE.dataset_folder[1:] + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now) + os.path.basename(dataset_NOCHANGE.dataset_folder) + '_{0:%m%d}_{0:%H%M}.pth'.format(now, now))
        
        if earlystopper == 1:
            write_gspread.update_gspread(dataset_NOCHANGE.dataset_folder, 'WRN', dataset_NOCHANGE.dataset_directory, now, train_acc, train_loss, test_acc, test_loss, epoch+1, learner.module.epochs, True, save_place, otherparams)
            break
        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    print("Start Training")
    print("")
    main(learner)

