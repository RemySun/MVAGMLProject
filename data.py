import scipy.io as sio
import numpy as np
import glob
import pickle

def load_text(path):
    raw_data = sio.loadmat(path)
    text_data = []
    for sample in raw_data['D1'][0]:
        text_data.append(sample[7][0,:])
    for sample in raw_data['D2'][0]:
        text_data.append(sample[7][0,:])
    return text_data

def load_labels(directory,n_samples,n_labels):
    filenames =  glob.glob(directory+"/*_trainval.txt")
    sample_names = {}
    for i,filename in enumerate(glob.glob("VOCdevkit/VOC2007/JPEGImages/train/*")):
        sample_names.update({int(filename.split('/')[-1].split('.')[0]):i})
    labels = np.zeros((n_samples,n_labels))
    for i,filename in enumerate(filenames):
        raw_data = np.genfromtxt(filename)
        for sample, ground_truth in raw_data:
            if ground_truth == 1:
                labels[sample_names[int(sample)],i]=1
    return labels

def load_imgs(path):
    return pickle.load(open(path,'rb'))
