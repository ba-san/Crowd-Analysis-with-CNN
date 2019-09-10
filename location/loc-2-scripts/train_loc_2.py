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
import random
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
from dataset4loc_2 import get_data
import dataset4loc_2
from tqdm import tqdm
from wide_resnet_loc_1 import WideResNet  # change here if you do transfer learning

from joblib import Parallel, delayed
import matplotlib.pyplot as plt

random.seed(32)
num_loc = 2  # must be more than 1

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
    hp = {"batch_size": 64,  #changed
          "lr": 1.0e-2,      #changed
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
    paths_total = []
    train_loss, n_train = 0, 0
    lr = optimizer.param_groups[0]["lr"]
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for batch_idx, (data, target, paths) in enumerate(train_data):
        data = data.to(device)
        
        loc = learner(data)

        #loc = loc.view(1, -1)
        loss_list = []
        for i in range(num_loc):
            loss_list.append((loss_func(loc[:, i*2], target[i][0].to(device).float()) + loss_func(loc[:, i*2+1], target[i][1].to(device).float())))

        #Euclidean loss (without sqrt)
        loss = sum(loss_list)/num_loc
        #loc = loc.view(-1, 1)

        loc_total_train.extend(loc)
        target_total_train.extend(target)
        paths_total.extend(paths)

        optimizer.zero_grad() # clears the gradients of all optimized tensors for next train.
        loss.backward()       # backpropagation, compute gradients
        optimizer.step()      # apply gradients. renew learning rate.

        train_loss += loss.item() * len(target[0]) # loss.item() is a loss num, but not a tensor. target.size(0) is batch size.
        n_train += len(target[0])

        bar.set_description("Loss(MSE): {0:.6f}".format(train_loss / n_train))
        
        bar.update()
    bar.close()
    
    #if loc_check==True:
    #if (epoch>1 and epoch%40==0) or epoch==199:
    if epoch==999:
        extract_num = 500
        hp_for_record= get_arguments()
        bs = hp_for_record["batch_size"]
        if os.path.exists(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/train-pred-{1}/'.format(now, epoch)):
            shutil.rmtree(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/train-pred-{1}/'.format(now, epoch))
        os.makedirs(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/train-pred-{1}/'.format(now, epoch))

        bar = tqdm(desc = "dotting", total = extract_num, leave = False)
        f = open(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/train-pred-{1}/'.format(now, epoch) + str(epoch) + '-pred-locations.txt', 'w')

        random_extract_list = random.sample(range(len(paths_total)), k=extract_num)

        for i in random_extract_list:
            path_separated = paths_total[i].split('/')
            img_lists = glob.glob(dataset4loc_2.full_path + '/train/*/*'.format(now, epoch))
            img_file = [s for s in img_lists if path_separated[-1] in s]
            img = cv2.imread(img_file[0])
            new_img_file = img_file[0].split('/')
            
            f.write('{}\n'.format(os.path.basename(img_file[0])))
            for k in range(num_loc):
                f.write('prediction  :{},{}\n'.format(int(loc_total_train[i][k*2]+0.5), int(loc_total_train[i][k*2+1]+0.5)))
            for k in range(num_loc):
                if k != num_loc-1:
                    f.write('ground truth:{},{}\n'.format(int(target_total_train[num_loc*int(i/bs)+k][0][i%bs]+0.5), int(target_total_train[num_loc*int(i/bs)+k][1][i%bs]+0.5)))
                else:
                    f.write('ground truth:{},{}\n\n'.format(int(target_total_train[num_loc*int(i/bs)+k][0][i%bs]+0.5), int(target_total_train[num_loc*int(i/bs)+k][1][i%bs]+0.5)))
            
            if 0<=int(loc_total_train[i][0])<img.shape[0] and 0<=int(loc_total_train[i][1])<img.shape[1] \
            and 0<=int(loc_total_train[i][2])<img.shape[0] and 0<=int(loc_total_train[i][3])<img.shape[1]:
                #colors = [[18, 0, 230],[0, 152, 243],[0, 241, 255],[31, 195, 143],[68, 153, 0],[150, 158, 0],[233, 160, 0],[183, 104, 0],[136, 32, 29]]
                colors = [[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230]]
                for k in range(num_loc):
                    img[int(loc_total_train[i][2*k+1]), int(loc_total_train[i][2*k])] = colors[k]
                    if int(loc_total_train[i][2*k+1])+1 < img.shape[1]:
                        img[int(loc_total_train[i][2*k+1])+1, int(loc_total_train[i][2*k])] = colors[k]
                    if int(loc_total_train[i][2*k+1])-1 >= 0:
                        img[int(loc_total_train[i][2*k+1])-1, int(loc_total_train[i][2*k])] = colors[k]
                    if int(loc_total_train[i][2*k])+1 < img.shape[0]:
                        img[int(loc_total_train[i][2*k+1]), int(loc_total_train[i][2*k])+1] = colors[k]
                    if int(loc_total_train[i][2*k])-1 >= 0:    
                        img[int(loc_total_train[i][2*k+1]), int(loc_total_train[i][2*k])-1] = colors[k]
            cv2.imwrite(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/train-pred-{1}/'.format(now, epoch) + new_img_file[-1], img)
            bar.update()
        bar.close()
        f.close()

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
        
        #loc = loc.view(1, -1)
        loss_list = []
        for i in range(num_loc):
            loss_list.append((loss_func(loc[:, i*2], target[i][0].to(device).float()) + loss_func(loc[:, i*2+1], target[i][1].to(device).float())))

        #Euclidean loss (without sqrt)
        loss = sum(loss_list)/num_loc
        #loc = loc.view(-1, 1)

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
        extract_num = 500
        hp_for_record= get_arguments()
        bs = hp_for_record["batch_size"]
        if os.path.exists(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch)):
            shutil.rmtree(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch))
        os.makedirs(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch))

        bar = tqdm(desc = "dotting", total = extract_num, leave = False)
        f = open(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch) + str(epoch) + '-pred-locations.txt', 'w')

        random_extract_list = random.sample(range(len(paths_total)), k=extract_num)

        for i in random_extract_list:
            path_separated = paths_total[i].split('/')
            img_lists = glob.glob(dataset4loc_2.full_path + '/test/*/*'.format(now, epoch))
            img_file = [s for s in img_lists if path_separated[-1] in s]
            img = cv2.imread(img_file[0])
            new_img_file = img_file[0].split('/')
            
            f.write('{}\n'.format(os.path.basename(img_file[0])))
            for k in range(num_loc):
                f.write('prediction  :{},{}\n'.format(int(loc_total_test[i][k*2]+0.5), int(loc_total_test[i][k*2+1]+0.5)))
            for k in range(num_loc):
                if k != num_loc-1:
                    f.write('ground truth:{},{}\n'.format(int(target_total_test[num_loc*int(i/bs)+k][0][i%bs]+0.5), int(target_total_test[num_loc*int(i/bs)+k][1][i%bs]+0.5)))
                else:
                    f.write('ground truth:{},{}\n\n'.format(int(target_total_test[num_loc*int(i/bs)+k][0][i%bs]+0.5), int(target_total_test[num_loc*int(i/bs)+k][1][i%bs]+0.5)))
            
            if 0<=int(loc_total_test[i][0])<img.shape[0] and 0<=int(loc_total_test[i][1])<img.shape[1] \
            and 0<=int(loc_total_test[i][2])<img.shape[0] and 0<=int(loc_total_test[i][3])<img.shape[1]:
                #colors = [[18, 0, 230],[0, 152, 243],[0, 241, 255],[31, 195, 143],[68, 153, 0],[150, 158, 0],[233, 160, 0],[183, 104, 0],[136, 32, 29]]
                colors = [[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230],[18, 0, 230]]
                for k in range(num_loc):
                    img[int(loc_total_test[i][2*k+1]), int(loc_total_test[i][2*k])] = colors[k]
                    if int(loc_total_test[i][2*k+1])+1 < img.shape[1]:
                        img[int(loc_total_test[i][2*k+1])+1, int(loc_total_test[i][2*k])] = colors[k]
                    if int(loc_total_test[i][2*k+1])-1 >= 0:
                        img[int(loc_total_test[i][2*k+1])-1, int(loc_total_test[i][2*k])] = colors[k]
                    if int(loc_total_test[i][2*k])+1 < img.shape[0]:
                        img[int(loc_total_test[i][2*k+1]), int(loc_total_test[i][2*k])+1] = colors[k]
                    if int(loc_total_test[i][2*k])-1 >= 0:    
                        img[int(loc_total_test[i][2*k+1]), int(loc_total_test[i][2*k])-1] = colors[k]
            cv2.imwrite(dataset4loc_2.full_path + '/log/{0:%m%d}_{0:%H%M}/test-pred-{1}/'.format(now, epoch) + new_img_file[-1], img)
            bar.update()
        bar.close()
        f.close()

    return test_loss / n_test


def main(learner):

    device = "cuda"
    
    ## load pretrained model ##
    global model
    model = "loc-1-extensive-extracted_output_x_x_x_x_x_resized_x_x_0820_2220_66.pth" # set model here
    learner.load_state_dict(torch.load('/mnt/CrowdData/dataset/resized/loc-1/loc-1-extensive-extracted_output_x_x_x_x_x_resized_x_x/log/0820_2220/models/' + model))
    learner.full_conn = nn.Linear(in_features=640, out_features=4, bias=True)
    learner.variance4pool = 12
    ###########################

    learner = torch.nn.DataParallel(learner, device_ids=[0, 1]) # make parallel

    train_data, test_data = get_data(learner.module.batch_size)
    
    global now
    now = datetime.datetime.now()
    if not os.path.exists(dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/models'.format(now)):
        os.makedirs(dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/models'.format(now))
        shutil.copyfile("./train_loc_2.py", dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/train_loc_2_{0:%m%d}_{0:%H%M}.py'.format(now))
        shutil.copyfile("./dataset4loc_2.py", dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/dataset4loc_2_{0:%m%d}_{0:%H%M}.py'.format(now))
    
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

    milestones = learner.module.lr_step
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.module.lr_decay)

    rsl_keys = ["lr", "epoch", "TrainLoss", "TestLoss", "Time"]
    rsl = []
    
    print_result(rsl_keys)
    
    global patience, counter, best_loss, epoch
    patience = 200 #set any adequate number here
    counter = 0
    earlystopper = 0
    best_loss = None
    
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
                torch.save(learner.module.state_dict(), dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/models/'.format(now, now) + os.path.basename(dataset4loc_2.dataset_folder) + '_{0:%m%d}_{0:%H%M}_{1}.pth'.format(now, epoch))
            ######################


        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_loss, test_loss, time_now])})
     
        draw_graph.draw_graph_regress(learner.module.epochs, epoch, train_loss, test_loss, os.path.basename(dataset4loc_2.dataset_folder), save_place=dataset4loc_2.dataset_directory  + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now), ymax=300.0)
        
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

        save_place = dataset4loc_2.dataset_directory + dataset4loc_2.dataset_folder + '/log/{0:%m%d}_{0:%H%M}/'.format(now, now)
        #write_gspread.update_gspread(dataset4loc_2.dataset_folder, 'WRN', dataset4loc_2.dataset_directory, now, 'N/A(regress)', train_loss, 'N/A(regress)', test_loss, epoch+1, learner.module.epochs, False, save_place, otherparams)
        
        print_result(rsl[-1].values())
        scheduler.step()
          
        if earlystopper == 1:
            #write_gspread.update_gspread(dataset4loc_2.dataset_folder, 'WRN', dataset4loc_2.dataset_directory, now, 'N/A(regress)', train_loss, 'N/A(regress)', test_loss, epoch+1, learner.module.epochs, True, save_place, otherparams)
            break

        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    learner = WideResNet(hyperparameters)
    
    print("Start Training")
    print("")
    main(learner)

