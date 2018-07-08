import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# shape is 207 by 61
# so we now save the first 60 cols in x since the last one is result y
dataset = pd.read_csv("D:\PycharmProjects\Multi-Layer-Perceptron-using-TensorFlow\DataSet\sonar.csv")
X = dataset[dataset.columns[0:60]].values
y = dataset[dataset.columns[60]].values  # end col
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# rock is 1 mine is 0


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


Y = one_hot_encode(y)
print(Y)
