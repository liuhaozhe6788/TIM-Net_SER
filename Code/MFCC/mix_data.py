import numpy as np
import os

CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral", "sad","surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad","surprise")#emovo
ESD_CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")#esd
INTERSECT_CLASS_LABELS = ("angry", "happy", "neutral", "sad","surprise")
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS,
               "ESD_train": ESD_CLASS_LABELS,
               "INTERSECT":INTERSECT_CLASS_LABELS}
CLASS_LABELS_OFFSETS = {"CASIA": [0, -1, 1, 2, 3, 4],
                        "EMODB": [0, -1, -1, -1, 1, 2, 3],
                        "SAVEE": [0, -1, -1, 1, 2, 3, 4], 
                        "RAVDE": [0, -1, -1, -1, 1, 2, 3, 4],
                        "IEMOCAP": [0, 1, 2, 3],
                        "EMOVO": [0, -1, -1, 1, 2, 3, 4],
                        "ESD_train": [0, 1, 2, 3, 4]}

if __name__ == '__main__':
    names = os.listdir("./MFCC/")
    tempx = []
    tempy = []
    MFCC_len_max = 0
    for name in names:
        if name.endswith('npy') and not name.endswith('test.npy') and not name == 'INTERSECT.npy':
            data = np.load("./MFCC/"+name, allow_pickle=True).item()
            if MFCC_len_max < data['x'].shape[1]:
                MFCC_len_max = data['x'].shape[1]
            idx = np.argmax(data["y"], axis=1)
            offsets = CLASS_LABELS_OFFSETS[name.split('.')[0]]
            datax=np.concatenate(
                [data['x'][np.where(idx==i)[0]] for i in range(len(offsets))  if offsets[i]!=-1], axis=0)
            datay=np.concatenate(
                [np.ones(idx[np.where(idx==i)[0]].shape) * offsets[i] for i in range(len(offsets))  if offsets[i]!=-1], axis=0).astype(int)
            tempx.append(datax)
            y = np.zeros((datax.shape[0], len(INTERSECT_CLASS_LABELS)))
            y[np.arange(datay.shape[0]), datay] = 1
            tempy.append(y)
    for idx in range(len(tempx)):
        pad_len = MFCC_len_max-tempx[idx].shape[1]
        pad_before = pad_len//2
        tempx[idx] = np.pad(tempx[idx], ((0, 0), (pad_before, pad_len-pad_before), (0, 0)))
    x_source = np.concatenate(tuple(tempx), axis=0)
    y_source = np.concatenate(tuple(tempy), axis=0)
    data = np.array({'x':x_source, 'y':y_source})
    y_ = np.argmax(y_source, axis=1)
    np.save('./MFCC/INTERSECT.npy', data)