import numpy
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
import pickle
from scipy import stats

import csv


datafolder = "tinyml_contest_data_training/"
train_idx_path= "data_indices/train_indice.csv"
test_idx_path = "data_indices/test_indice.csv"


def get_label(filename):
    names = str.split(filename, '-')
    return names[1]

def gen_dataset_pre_idx(datapath):

    def get_data_with_idx(idx_path):
        df_idx = pd.read_csv(idx_path)
        labels = []
        data   = []
        for i, row in df_idx.iterrows():
            # print(row["label"], row["Filename"])
            dataf = os.path.join(datapath, row["Filename"])
            if os.path.isfile(dataf):
                d = numpy.loadtxt(dataf)
                label = row["label"]
                label_name = get_label(dataf)
                # print(dataf, label, len(d))
                labels.append(label)
                data.append(d)
        return labels, data

    labels, data = get_data_with_idx(test_idx_path)
    print(labels, len(data))

    with open("test_data.pkg", "wb") as fp:  # Pickling
        pickle.dump(data, fp)
    with open("test_label.pkg", "wb") as fp:  # Pickling
        pickle.dump(labels, fp)

    labels, data = get_data_with_idx(train_idx_path)
    print(labels, len(data))

    with open("train_data.pkg", "wb") as fp:  # Pickling
        pickle.dump(data, fp)
    with open("train_label.pkg", "wb") as fp:  # Pickling
        pickle.dump(labels, fp)
    return

def gen_dataset(path):
    labels = []
    data  = []


    file = os.listdir(path)
    for f in file:
        dataf = os.path.join(path, f)
        if os.path.isfile(dataf):
            d = numpy.loadtxt(dataf)
            label = get_label(dataf)
            print(label, len(d))

            labels.append(label)
            data.append(d)

    with open("dataset.pkg", "wb") as fp:  # Pickling
        pickle.dump([labels, data], fp)
    return

if __name__ == '__main__':
    #gen_dataset(datafolder)

    gen_dataset_pre_idx(datafolder)
    pass
