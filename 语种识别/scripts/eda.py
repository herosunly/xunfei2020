#%%
import librosa
import sklearn
import numpy as np
import pandas as pd 
from conf import config
import matplotlib.pyplot as plt
import tqdm

train_df = pd.read_csv(config.train_path)

# %%
wavelen = []
bad_files = []
for i, filename in tqdm.tqdm(enumerate(train_df.filename.values)):
    try:
        x, sr = librosa.load(filename)
        wavelen.append(len(x))
    except BaseException:
        bad_files.append(filename)

# %%
train_df['wavelen'] = wavelen
train_df.wavelen.describe()

# %%
src = '/home/gongxj/students/usera/houys/project/lan/input/train/L004-sksk/train/L004-train-1.wav'
librosa.load(src)

# %%
bad_files = []
import glob  
for f in glob.glob('/home/gongxj/students/usera/houys/project/lan/input/train/L004-sksk/train/*.wav'):
    try:
        x, sr = librosa.load(f)
    except BaseException:
        bad_files.append(f)

# %%
bad_files = pd.DataFrame(bad_files)
bad_files.to_csv('bad_files.csv', index = None)

# %%
bad_files.values

# %%
