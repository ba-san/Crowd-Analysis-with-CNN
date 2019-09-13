import os
import cv2
import glob
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

csv_path = glob.glob('./csv/*/*.csv')

num_loc = 9

def cropping(csv):
	df = pd.read_csv(csv, index_col=0)
	bar = tqdm(desc = "Processing", total = int(len(df)/num_loc), leave = False)
	for i in range(int(len(df)/num_loc)):
		if df['y'][num_loc*i]>41 or df['y'][num_loc*i]<1 or df['x'][num_loc*i]>46 or df['x'][num_loc*i]<2 \
		or df['y'][num_loc*i+1]>41 or df['y'][num_loc*i+1]<1 or df['x'][num_loc*i+1]>46 or df['x'][num_loc*i+1]<2 \
		or df['y'][num_loc*i+2]>41 or df['y'][num_loc*i+2]<1 or df['x'][num_loc*i+2]>46 or df['x'][num_loc*i+2]<2 \
		or df['y'][num_loc*i+3]>41 or df['y'][num_loc*i+3]<1 or df['x'][num_loc*i+3]>46 or df['x'][num_loc*i+3]<2 \
		or df['y'][num_loc*i+4]>41 or df['y'][num_loc*i+4]<1 or df['x'][num_loc*i+4]>46 or df['x'][num_loc*i+4]<2 \
		or df['y'][num_loc*i+5]>41 or df['y'][num_loc*i+5]<1 or df['x'][num_loc*i+5]>46 or df['x'][num_loc*i+5]<2 \
		or df['y'][num_loc*i+6]>41 or df['y'][num_loc*i+6]<1 or df['x'][num_loc*i+6]>46 or df['x'][num_loc*i+6]<2 \
		or df['y'][num_loc*i+7]>41 or df['y'][num_loc*i+7]<1 or df['x'][num_loc*i+7]>46 or df['x'][num_loc*i+7]<2 \
		or df['y'][num_loc*i+8]>41 or df['y'][num_loc*i+8]<1 or df['x'][num_loc*i+8]>46 or df['x'][num_loc*i+8]<2:
			img_path = df['image'][num_loc*i]
			img_name = os.path.basename(img_path)
			before_comma = img_name.split('.')[0] + '.' + img_name.split('.')[1]
			after_comma = img_name.split('.')[2]
			delete_path = glob.glob('./*/*/' + before_comma + '_resized.' + after_comma)
			os.remove(delete_path[0])
			df = df[~df['image'].str.contains(img_path)]
		bar.update()
	bar.close()
	df = df.reset_index(drop=True)
	df.to_csv(csv)

if __name__ == '__main__':
	
	result = Parallel(n_jobs=-1)([delayed(cropping)(csv) for csv in csv_path])
