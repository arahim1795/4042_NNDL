#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import random
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# * Parameters
# - input: LB to Tendency
FEATURE_INPUT = 21
# - NSP = 1, 2, 3
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 100000
batch_size = [4,8,16,32,64]
num_neurons = 10
seed = 10
np.random.seed(seed)
decay = math.pow(10, -6)

# * Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X__, Y__ = [], []
X_, Y_ = [], []
batch_X, batch_Y = [], []

# Function for creating batches
# inputs are the inputs, outputs and batch size
# creates (num_items/batch_size) batches each containing batch_size items
# except for last one: size of (num_items % batch_size)


# def create_batch(X_, Y_, batch_size):
#     batch_X, batch_Y = [], []
#     for i in range(1):
#         for j in range(0, len(X_[i]), batch_size):
#             if j + batch_size < len(X_[i]):
#                 batch_X.append(X_[i][j:j+batch_size])
#                 batch_Y.append(Y_[i][j:j+batch_size])
#             else:
#                 batch_X.append(X_[i][j:len(X_[i])])
#                 batch_Y.append(Y_[i][j:len(Y_[i])])

#     return batch_X, batch_Y

# # * load folds
dataset = []
for i in range(5):
    filename = 'input/A/fold_' + str(i) + '.csv'
    dataset.append(np.genfromtxt(filename, delimiter=','))


# * form one-left-out processed datasets
def leave_one_out_dataset(k_value, data):
    # create data structure
    leave_one_out = [None] * k_value
    for i in range(k_value):
        leave_one_out[i] = [None] * 2
        for j in range(2):
            leave_one_out[i][j] = [None] * 2

    # add data to structure
    for i in range(k_value):
        leave_one_out[i][0][0] = np.zeros(shape=(1, 23))
        leave_one_out[i][1][0] = np.zeros(shape=(1, 23))
        for j in range(k_value):
            if i == j:
                leave_one_out[i][1][0] = np.concatenate(
                    (leave_one_out[i][1][0], data[j])
                    )
                continue
            leave_one_out[i][0][0] = np.concatenate((leave_one_out[i][0][0], data[j]))
    
    # remove placeholder entry
    for i in range(k_value):
        leave_one_out[i][0][0] = leave_one_out[i][0][0][1:]
        leave_one_out[i][1][0] = leave_one_out[i][1][0][1:]

    # process data within structure
    for i in range(k_value):
        for j in range(2):
            temp_var = leave_one_out[i][j][0]
            leave_one_out[i][j][0] = temp_var[:, :FEATURE_INPUT]
            leave_one_out[i][j][0] = scale(
                leave_one_out[i][j][0],
                np.min(leave_one_out[i][j][0], axis=0),
                np.max(leave_one_out[i][j][0], axis=0)
            )
            temp_var = temp_var[:, -1].astype(int)
            Y_one_hot = np.zeros((temp_var.shape[0], NUM_CLASSES))
            Y_one_hot[np.arange(temp_var.shape[0]), temp_var-1] = 1
            leave_one_out[i][j][1] = Y_one_hot

    return leave_one_out

test = leave_one_out_dataset(5, dataset)

# batch_X, batch_Y = create_batch(X_[0], Y_[0], 4)

# # * Graph Start
# # Create the model
# x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
# y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# # Hidden Layer
# layer_1_weights = tf.Variable(tf.truncated_normal(
#     [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
# layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
# layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

# layer_1_output = tf.nn.relu(layer_1_var)

# # Softmax
# layer_final_weights = tf.Variable(tf.truncated_normal(
#     [num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
# layer_final_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
# logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

# # Regularisation (L2)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
#     labels=y_, logits=logits)
# loss = tf.reduce_mean(cross_entropy)

# loss = tf.reduce_mean(loss + (decay*tf.nn.l2_loss(logits)))

# # Minimising Loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train_op = optimizer.minimize(loss)

# correct_prediction = tf.cast(
#     tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
# accuracy = tf.reduce_mean(correct_prediction)

# train_acc_set, loss_set, test_acc_set = [], [], []

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     train_acc, test_acc = [], []
#     for j in range(epochs):
#         for i in range(len(batch_X)):
#            # Batch train
#             for start, end in zip(range(0, len(X_[0]), batch_size), range(batch_size, len(X_[0]), batch_size)):
#                 if start+batch_size < len(X_[0]):
#                     train_op.run(feed_dict={x: X_[0][start:end], y_: Y_[0][start:end]})
#                 else: 
#                     train_op.run(feed_dict={x: X_[0][start:len(X_[0])], y_: Y_[0][start:len(Y_[0])]})
#             # evalutation
#             train_acc.append(accuracy.eval(feed_dict={x: batch_X[i], y_: batch_Y[i]}))
#             # test_acc.append(accuracy.eval(feed_dict={x: X_[1], y_: Y_[1]}))
#             if j%1000 == 0:
#                 print('iter %d: tr-acc %g, te-acc %g' % (j, train_acc[j], test_acc[j]))
#     train_acc_set.append(train_acc)
#     test_acc_set.append(test_acc)
# print(train_acc_set)
# print('-')
# print(test_acc_set)

# # plot learning curves
# plt.figure(1)
# plt.plot(range(epochs), train_acc, label ='Train Accuracy')
# plt.plot(range(epochs), test_acc, label = 'Test Accuracy')
# plt.xlabel(str(epochs) + ' iterations')
# plt.ylabel('Train/Test accuracy')
# plt.legend()
# plt.show()
