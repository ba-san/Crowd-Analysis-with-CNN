import os
import glob
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

num_loc = 7

PWD = os.getcwd()

csvs = glob.glob('./csv/*/*')

def raster(category): 
	
	df = pd.read_csv(category, index_col=0)
	
	csv_num = int(len(df)/num_loc)
	
	bar = tqdm(desc = str(category), total = csv_num, leave = False)
	for i in range(csv_num):
		for j in range(num_loc-1): ##bubble sort
			for k in range(num_loc-1, j, -1):
				if df.loc[num_loc*i+k-1, 'y']>df.loc[num_loc*i+k, 'y']:
					temp_x = df.loc[num_loc*i+k-1, 'x']
					temp_y = df.loc[num_loc*i+k-1, 'y']
					df.loc[num_loc*i+k-1, 'x'] = df.loc[num_loc*i+k, 'x']
					df.loc[num_loc*i+k-1, 'y'] = df.loc[num_loc*i+k, 'y']
					df.loc[num_loc*i+k, 'x'] = temp_x
					df.loc[num_loc*i+k, 'y'] = temp_y
				
				elif df.loc[num_loc*i+k-1, 'y']==df.loc[num_loc*i+k, 'y']:
					if df.loc[num_loc*i+k-1, 'x']>df.loc[num_loc*i+k, 'x']:
						temp_y = df.loc[num_loc*i+k-1, 'x']
						df.loc[num_loc*i+k-1, 'x'] = df.loc[num_loc*i+k, 'x']
						df.loc[num_loc*i+k, 'x'] = temp_y
		bar.update()
	bar.close()

	df.to_csv(category)
	print('category:{} finished.'.format(category))


if __name__ == '__main__':
	
	result = Parallel(n_jobs=-1)([delayed(raster)(category) for category in csvs])
