import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_dataset():
    # shape is 207 by 61
    # so we now save the first 60 cols in x since the last one is result y
    dataset = pd.read_csv("DataSet/sonar.csv")
    X = dataset[dataset.columns[0:60]].values
    y = dataset[dataset.columns[60]].values  # end col
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return (X, Y)

# this is speperating the class to like rock will be 10 and mine 01
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# first we load the data
X, Y = read_dataset()

# shuffel it
X, Y = shuffle(X, Y, random_state=1)
print(shuffle(X, Y, random_state=1))
# split the test cases and train
train_x, test_x, train_y, train_y = train_test_split(X, Y, test_size=0.2, random_state=1)

# testing the shape
# print(train_x.shape, test_x.shape, train_y.shape, train_y.shape)

# defining the imp parameters
