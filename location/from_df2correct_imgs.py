
import os
import cv2
from tqdm import tqdm
import pandas as pd

csv_path = './df_test.csv'
dst_path = './test_answers/'

df = pd.read_csv(csv_path, index_col=0)

os.makedirs(dst_path)

bar = tqdm(desc = "Processing", total = len(df), leave = False)
for i in range(len(df)):
	if os.path.exists(dst_path  + df['file'][i]):
		ori = cv2.imread(dst_path  + df['file'][i])
	else:
		ori = cv2.imread(df['path'][i])
	ori[min(int(df['y'][i]-1), ori.shape[0]-1), min(int(df['x'][i]-1), ori.shape[0]-1)] = [255,255,255]
	ori[min(int(df['y'][i]), ori.shape[0]-1), min(int(df['x'][i]-1), ori.shape[0]-1)] = [255,255,255]
	ori[min(int(df['y'][i]-1), ori.shape[0]-1), min(int(df['x'][i]), ori.shape[0]-1)] = [255,255,255]
	ori[min(int(df['y'][i]-2), ori.shape[0]-1), min(int(df['x'][i]-1), ori.shape[0]-1)] = [255,255,255]
	ori[min(int(df['y'][i]-1), ori.shape[0]-1), min(int(df['x'][i]-2), ori.shape[0]-1)] = [255,255,255]
	cv2.imwrite(dst_path  + df['file'][i], ori)
	bar.update()
bar.close()
