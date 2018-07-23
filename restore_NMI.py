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



def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# first we load the data
X, Y = read_dataset()

# shuffel it
# X, Y = shuffle(X, Y, random_state=1)
# print(shuffle(X, Y, random_state=1))
# split the test cases and train
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)

learning_rate = 0.3
training_epocs = 1000            # no of iterations
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
# print("n_dim", n_dim)
n_classes = 2       #no of classes Mine and Rock so 2
model_path = "D:\\PycharmProjects\\TensorFlow_Models\\NMI"

# defining the hidden layer and ip and op layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60


x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
y_ = tf.placeholder(tf.float32, [None, n_classes])


# defining my model
def multilayer_perceptron(x, weights,biases):
    #hidden layer with sigmoid activation

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)

    # hidden layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)

    # hidden layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)

    # hidden layer 4
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #output layer
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer


weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes])),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes])),
}

y = multilayer_perceptron(x, weights, biases)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess,model_path)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction_run = sess.run(prediction, feed_dict={x: X[5].reshape(1,60)})

print(prediction_run)

