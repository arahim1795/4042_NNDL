#
# Project 1, starter code part a
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
FEATURE_INPUT = 21
# - NSP = 1, 2, 3
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 100000
num_neurons = 10
seed = 10
np.random.seed(seed)
decay = math.pow(10, -6)

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []

data = np.genfromtxt('input/A/train_a_data.csv', delimiter=',')
# process X
X_.append(data[:, :FEATURE_INPUT])
X_[0] = scale(X_[0], np.min(X_[0], axis=0), np.max(X_[0], axis=0))
# process Y
Y_temp = data[:, -1].astype(int)
Y_one_hot = np.zeros((Y_temp.shape[0], NUM_CLASSES))
Y_one_hot[np.arange(Y_temp.shape[0]), Y_temp-1] = 1
Y_.append(Y_one_hot)

data = np.genfromtxt('input/A/test_a_data.csv', delimiter=',')
# process X
X_.append(data[:, :FEATURE_INPUT])
X_[1] = scale(X_[1], np.min(X_[1], axis=0), np.max(X_[1], axis=0))
# process Y
Y_temp = data[:, -1].astype(int)
Y_one_hot = np.zeros((Y_temp.shape[0], NUM_CLASSES))
Y_one_hot[np.arange(Y_temp.shape[0]), Y_temp-1] = 1
Y_.append(Y_one_hot)


# for Qn
# - experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

# n = trainX.shape[0]

# Graph Start

# Create the model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Hidden Layer 1
# Truncated normal 1st variable is num of inputs, 2nd layer is num of outputs, std dev is over the total number of inputs
layer_1_weights = tf.Variable(tf.truncated_normal(
    [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

layer_1_output = tf.nn.relu(layer_1_var)

# Hidden Layer 2 is also the same relu 
# Takes in layer 1 output
layer_2_weights = tf.Variable(tf.truncated_normal(
    [num_neurons,num_neurons], stddev=1.0/math.sqrt(float(num_neurons))),name = "two_weights")
layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name = 'two_biases')
layer_2_var = tf.matmul(layer_1_output, layer_2_weights) +  layer_2_biases

layer_2_output = tf.nn.relu(layer_2_var)
# Softmax
layer_final_weights = tf.Variable(tf.truncated_normal(
    [num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
layer_final_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
logits = tf.matmul(layer_2_output, layer_final_weights) + layer_final_biases

# Regularisation (L2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y_, logits=logits)
loss = tf.reduce_mean(cross_entropy)

loss = tf.reduce_mean(loss + (decay*tf.nn.l2_loss(logits)))

# Minimising Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_acc_set, loss_set, test_acc_set = [], [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc, test_acc = [], []
    for j in range(epochs):
        # train
        train_op.run(feed_dict={x: X_[0], y_: Y_[0]})
        # evalutation
        train_acc.append(accuracy.eval(feed_dict={x: X_[0], y_: Y_[0]}))
        test_acc.append(accuracy.eval(feed_dict={x: X_[1], y_: Y_[1]}))
        if j%1000 == 0:
            print('iter %d: tr-acc %g, te-acc %g' % (j, train_acc[j], test_acc[j]))
    train_acc_set.append(train_acc)
    test_acc_set.append(test_acc)
# print(train_acc_set)
# print('-')
# print(test_acc_set)

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc, label ='Train Accuracy')
plt.plot(range(epochs), test_acc, label = 'Test Accuracy')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train/Test accuracy')
plt.legend()
plt.show()