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
for i, filename in tqdm.tqdm(enumerate(train_df.filename.values)):
    x, sr = librosa.load(filename)
    wavelen.append(len(x))
train_df['wavelen'] = wavelen
# %%
train_df.wavelen.describe()

# %%
