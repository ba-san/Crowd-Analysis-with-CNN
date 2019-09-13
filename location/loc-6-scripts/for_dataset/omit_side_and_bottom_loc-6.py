import os
import cv2
import glob
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

csv_path = glob.glob('./csv/*/*.csv')

def cropping(csv):
	df = pd.read_csv(csv, index_col=0)
	bar = tqdm(desc = "Processing", total = int(len(df)/6), leave = False)
	for i in range(int(len(df)/6)):
		if df['y'][6*i]>38 or df['y'][6*i]<2 or df['x'][6*i]>43 or df['x'][6*i]<5 \
		or df['y'][6*i+1]>38 or df['y'][6*i+1]<2 or df['x'][6*i+1]>43 or df['x'][6*i+1]<5 \
		or df['y'][6*i+2]>38 or df['y'][6*i+2]<2 or df['x'][6*i+2]>43 or df['x'][6*i+2]<5 \
		or df['y'][6*i+3]>38 or df['y'][6*i+3]<2 or df['x'][6*i+3]>43 or df['x'][6*i+3]<5 \
		or df['y'][6*i+4]>38 or df['y'][6*i+4]<2 or df['x'][6*i+4]>43 or df['x'][6*i+4]<5 \
		or df['y'][6*i+5]>38 or df['y'][6*i+5]<2 or df['x'][6*i+5]>43 or df['x'][6*i+5]<5:
			img_path = df['image'][6*i]
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
