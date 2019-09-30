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

# Variables

# - input: LB to Tendency
FEATURE_INPUT = 21
# - NSP = 1, 2, 3
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
batch_size = 32
num_neurons = 10
seed = 10
np.random.seed(seed)
decay = math.pow(10, -6)

# Data Pre-Processing
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter=',')
# - trainX is the input
# - train_Y is the output (NSP)
trainX = train_input[1:, :FEATURE_INPUT]
train_Y = train_input[1:, -1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
# - one hot matrix
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1

# - experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

n = trainX.shape[0]

# Graph Start

# Create the model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Hidden Layer
layer_1_weights = tf.Variable(tf.truncated_normal(
    [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

layer_1_output = tf.nn.relu(layer_1_var)

# Softmax
layer_final_weights = tf.Variable(tf.truncated_normal(
    [num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
layer_final_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y_, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    for i in range(epochs):
        train_op.run(feed_dict={x: trainX, y_: trainY})
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g' % (i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()