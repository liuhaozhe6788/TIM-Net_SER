import numpy as np
import os
import pathlib
from tqdm import tqdm

from utils import get_mfcc



ESD_CLASS_LABELS = ("Angry", "Happy", "Neutral", "Sad", "Surprise")

dataset_dir = "data/Emotional_Speech_Dataset"

X = []
y = []

for path in tqdm(pathlib.Path(dataset_dir).glob('**/train/*.wav')):
    X.append(get_mfcc(str(path), mean_signal_length=100000))
    y.append(ESD_CLASS_LABELS.index(path.parent.parent.name))

x_source = np.array(X)
y_ = np.array(y)

y_source= np.zeros((y_.size, y_.max() + 1))
y_source[np.arange(y_.size), y_] = 1

data = np.array({'x':x_source, 'y':y_source})
np.save('MFCC/ESD_train.npy', data)

X = []
y = []

for path in tqdm(pathlib.Path(dataset_dir).glob('**/test/*.wav')):
    X.append(get_mfcc(str(path), mean_signal_length=100000))
    y.append(ESD_CLASS_LABELS.index(path.parent.parent.name))

x_source = np.array(X)
y_ = np.array(y)

y_source= np.zeros((y_.size, y_.max() + 1))
y_source[np.arange(y_.size), y_] = 1

data = np.array({'x':x_source, 'y':y_source})
np.save('MFCC/ESD_test.npy', data)
