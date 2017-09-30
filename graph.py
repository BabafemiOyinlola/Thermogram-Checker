import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_features(txt_file):
    file = open(txt_file, 'r')
    features = file.readlines()
    features = sorted([int(feature.strip()) for feature in features])
    return features


def plot_graph():
    fts = read_features('text-files/normal-features-train.txt')
    fts_count = {}
    for feature in fts:
        if feature in fts_count:
            fts_count[feature] += 1
        else:
            fts_count[feature] = 1
    ft_num = []
    ft_freq = []
    for key, value in fts_count.items():
        ft_num.append(key)
        ft_freq.append(value)

    print(ft_num)
    print(ft_freq)
    plt.hist(fts, bins=30, range=(0, 5), color='red', histtype='bar', rwidth=0.8)
    plt.show()
    return fts_count

f = read_features('text-files/normal-features-train.txt')
print(f)
dict = plot_graph()
print(dict)