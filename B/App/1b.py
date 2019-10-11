#
# Project 1, starter code part b
#
import random
import math
import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(data):
    data_scaled = (data- np.mean(data, axis=0))/ np.std(data, axis=0)
    return data_scaled 

# Parameters
# - input: LB to Tendency
FEATURE_INPUT = 8

learning_rate = math.pow(10,-3)
epochs = 1000
num_neurons = 10
batch_size = 8
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []

data = np.genfromtxt('../Data/train_data.csv', delimiter=',')
# process X and Y
X_temp, Y_temp = data[:,:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)

data = np.genfromtxt('../Data/test_data.csv', delimiter=',')
#process X and Y
X_temp, Y_temp = data[:,:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)

idx = np.arange(X_[1].shape[0])
np.random.shuffle(idx)
prediciton = X_[1][idx]
actual = Y_[1][idx] 
actual_set = np.squeeze(np.asarray(actual))
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
loss = tf.reduce_mean(tf.square(y_ - logits))
regularizer = tf.nn.l2_loss(logits)

# Minimising Lossz`
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_loss_set,test_loss_set = [], []
prediction_set = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_error_set, test_error_set = [], []

    for i in range(epochs):
        # Batch
        for start, end in zip(range(0, len(X_[0]), batch_size), range(batch_size, len(X_[0]), batch_size)):
            if start+batch_size < len(X_[0]):
                train_op.run(feed_dict={x: X_[0][start:end], y_: Y_[0][start:end]})
            else: 
                train_op.run(feed_dict={x: X_[0][start:len(X_[0])], y_: Y_[0][start:len(Y_[0])]})
        # calculate loss
        train_error = loss.eval(feed_dict={x: X_[0], y_: Y_[0]})
        train_error_set.append(train_error)
        test_error = loss.eval(feed_dict={x: X_[1], y_: Y_[1]})
        test_error_set.append(test_error)
    for j in range (50):
    #predictions
        prediction_set.append(sess.run(logits,feed_dict={x:prediciton }))


# print(train_acc_set)
# print('-')
# print(test_acc_set)

# plot learning curves
# plt.figure(1)
# plt.plot(range(epochs), train_error_set, label ='Train Loss')
# plt.plot(range(epochs), test_error_set, label = 'Test Loss')
# plt.xlabel(str(epochs) + ' iterations')
# plt.ylabel('Train/Test Loss')
# plt.legend()
# plt.show()
print (prediction_set)
print(len(prediction_set))
plt.figure(1)
plt.scatter(50, prediction_set)
plt.plot(50, prediction_set, label="prediction")
plt.scatter(50, actual_set)
plt.plot(50, actual_set, label="actual")
plt.legend()
plt.show()