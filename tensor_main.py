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
    y = dataset[dataset.columns[60]].values
