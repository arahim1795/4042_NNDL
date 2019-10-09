#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# Parameters
# - input: LB to Tendency
FEATURE_INPUT = 8

learning_rate = math.pow(10,-3)
epochs = 100000
num_neurons = 10
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []

data = np.genfromtxt('input/B/train_b_data.csv', delimiter=',')
# process X and Y
X_temp, Y_temp = data[:,:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)

data = np.genfromtxt('input/B/test_b_data.csv', delimiter=',')
#process X and Y
X_temp, Y_temp = data[:,:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)


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

# Minimising Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_loss_set,test_loss_set = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_error_set, test_error_set = [], []
    for i in range(epochs):
        train_op.run(feed_dict={x: X_[0], y_: Y_[0]})
        train_error = loss.eval(feed_dict={x: X_[0], y_: Y_[0]})
        train_error_set.append(train_error)
        test_error = loss.eval(feed_dict={x: X_[1], y_: Y_[1]})
        test_error_set.append(test_error)
        if i % 100 == 0:
            print('iter %d: train error %g'%(i, train_error_set[i]))
            print('iter %d: test error %g'%(i, test_error_set[i]))
# print(train_acc_set)
# print('-')
# print(test_acc_set)

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_error_set, label ='Train Loss')
plt.plot(range(epochs), test_error_set, label = 'Test Loss')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train/Test Loss')
plt.legend()
plt.show()
