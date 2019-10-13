#
# Project 1, starter code part b
#
import random
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(data):
    data_scaled = (data- np.mean(data, axis=0))/ np.std(data, axis=0)
    return data_scaled 

# Parameters
# - input: LB to Tendency
FEATURE_INPUT = 5

learning_rate = math.pow(10,-3)
epochs = 10000
num_neurons = 10
batch_size = 8
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []
train_error_set,test_error_set = [],[]
count = 0
data = np.genfromtxt('../Data/train_data.csv', delimiter=',')
# process X and Y
X_temp, Y_temp = data[:,1:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

# Remove 2 random variable
for i in range(7):
    # remove 1 column
    # np delete removes the n row/column(depends on 3rd item)
    X_remove1 = np.delete(X_temp,i,1)
    train_error_set.append([])
    for j in range(6):
        X_remove2 = np.delete(X_remove1,j,1)
        X_.append(X_remove2)
        Y_.append(Y_temp)
        count +=1
        train_error_set[i].append([])

data = np.genfromtxt('../Data/test_data.csv', delimiter=',')
#process X and Y
X_temp, Y_temp = data[:,1:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

# Remove 2 random variable
for i in range(7):
    # remove 1 column
    # np delete removes the n row/column(depends on 3rd item)
    X_remove1 = np.delete(X_temp,i,1)
    test_error_set.append([])
    for j in range(6):
        X_remove2 = np.delete(X_remove1,j,1)
        X_.append(X_remove2)
        Y_.append(Y_temp)
        count +=1
        test_error_set[i].append([])

 
# for Qn
# - experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

# n = trainX.shape[0]

# Graph Start

# Create the model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, 1])

# Hidden Layer
layer_1_weights = tf.Variable(tf.truncated_normal(
    [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

layer_1_output = tf.nn.relu(layer_1_var)

# Final layer
layer_final_weights = tf.Variable(tf.truncated_normal(
    [num_neurons, 1], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
layer_final_biases = tf.Variable(tf.zeros([1]), name='final_biases')
logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

# Regularisation (L2)
regularization = tf.nn.l2_loss(layer_1_weights) + tf.nn.l2_loss(layer_final_weights)
loss = tf.reduce_mean(tf.square(y_ - logits))
l2_loss = tf.reduce_mean(loss + decay*regularization)

# Minimising Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     train_error_set,test_error_set = [[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[]]
#     for i in tqdm(range(epochs)):
#         for j in range(8):
#             # Batch
#             for start, end in zip(range(0, len(X_[j]), batch_size), range(batch_size, len(X_[j]), batch_size)):
#                 if start+batch_size < len(X_[j]):
#                     train_op.run(feed_dict={x: X_[j][start:end], y_: Y_[j][start:end]})
#                 else: 
#                     train_op.run(feed_dict={x: X_[j][start:len(X_[j])], y_: Y_[j][start:len(Y_[j])]})
#             # calculate loss
#             train_error = loss.eval(feed_dict={x: X_[j], y_: Y_[j]})
#             train_error_set[j].append(train_error)
#             test_error = loss.eval(feed_dict={x: X_[j], y_: Y_[j]})
#             test_error_set[j].append(test_error)

for i in range (7):
    for j in range(6):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for k in tqdm(range(epochs)):
                # Batch
                for start, end in zip(range(0, len(X_[i]), batch_size), range(batch_size, len(X_[i]), batch_size)):
                    if start+batch_size < len(X_[i]):
                        train_op.run(feed_dict={x: X_[i][start:end], y_: Y_[i][start:end]})
                    else: 
                        train_op.run(feed_dict={x: X_[i][start:len(X_[i])], y_: Y_[i][start:len(Y_[i])]})
                # calculate loss
                train_error_set[i][j].append(loss.eval(feed_dict={x: X_[i], y_: Y_[i]}))
                test_error_set[i][j].append(loss.eval(feed_dict={x: X_[i+42], y_: Y_[i+42]}))
# print(train_acc_set)
# print('-')
# print(test_acc_set)

plt.figure(1)
# plot learning curves
for i in range(7):
    for j in range(6):
        plt.plot(range(epochs), train_error_set[i][j], label ='Train Loss on removing column'+str(i) + "/"+str(j))
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train Loss')
plt.legend()

plt.figure(2)
for i in range(7):
    for j in range(6):
        plt.plot(range(epochs), test_error_set[i][j], label = 'Test Loss on removing column'+str(i) + "/"+str(j))
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test Loss')
plt.legend()
plt.show()
