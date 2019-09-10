# https://qiita.com/akabei/items/0eac37cb852ad476c6b9

import os 
import glob
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

global clear_last
clear_last = 1

def update_gspread(dset_fname, network_name, dataset_directory, tstamp, train_acc, train_loss, test_acc, test_loss, current_epoch, all_epoch, early_stopped=False, save_place=None, otherparams=None):
	global clear_last 
	
	### open google spreadsheet ###
	scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

	credentials = ServiceAccountCredentials.from_json_keyfile_name('something.json', scope)
	gc = gspread.authorize(credentials)
	final_wks = gc.open('sheetname').worksheet("Final")
	last_wks = gc.open('sheetname').worksheet("Last")


	### updating ###
	separate_dset_fname = dset_fname.split("_")
	dataset_name = os.path.basename(separate_dset_fname[0])
	crp_x_length = separate_dset_fname[-8]
	crp_y_length = separate_dset_fname[-7]
	x_interval = separate_dset_fname[-6]
	y_interval = separate_dset_fname[-5]
	thorn = separate_dset_fname[-4]
	
	num_of_class = len(os.listdir(dataset_directory + "/" + dset_fname + "/train/"))
	
	resize_x_length = separate_dset_fname[-2]
	resize_y_length = separate_dset_fname[-1]
	
	train_list = glob.glob(dataset_directory + "/" + dset_fname + "/train/*/*", recursive=True)
	test_list = glob.glob(dataset_directory + "/" + dset_fname + "/test/*/*", recursive=True)
		
	
	new_row = [dataset_name, network_name, dataset_directory + "/" + dset_fname, train_loss, test_loss, train_acc, test_acc,
				len(train_list)+len(test_list), len(train_list), len(test_list), num_of_class,
				crp_x_length, crp_y_length, resize_x_length, resize_y_length, x_interval, y_interval, thorn, current_epoch]
				
	for i in range(len(otherparams)):
		new_row.append(otherparams[i])
		
		
	if clear_last ==1:
		last_wks.resize(3)
		clear_last = 0
		csvcolumns = ['dataset name', 	'network name', 'place', 'train_loss', 'test_loss',	'train_acc', 'test_acc',
			'total img', 'train img', 'test img', 'num of class', 'x-side', 'y-side', 'x-side', 'y-side', 'x-interval', 'y-interval', 'thorn', 'epoch']
			
		for i in range(len(otherparams)):
			csvcolumns.append('from nw')
			
		df = pd.DataFrame(columns=csvcolumns)	
		df.to_csv(save_place + dataset_name + '_' + network_name + '_{0:%m%d}_{0:%H%M}'.format(tstamp, tstamp) + '_result.csv')
		
	last_wks.append_row(new_row)
				
	df = pd.read_csv(save_place + dataset_name + '_' + network_name + '_{0:%m%d}_{0:%H%M}'.format(tstamp, tstamp) + '_result.csv', index_col=0)
	series = pd.Series(new_row, index=df.columns)
	df = df.append(series, ignore_index=True)
	df.to_csv(save_place + dataset_name + '_' + network_name + '_{0:%m%d}_{0:%H%M}'.format(tstamp, tstamp) + '_result.csv')
	

	
	if current_epoch == all_epoch-1 or early_stopped == True:
		final_wks.append_row(new_row)
