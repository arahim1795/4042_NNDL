import math
import random

# read input from csv file and convert to row and columns
fid = open("01\ctg_data_cleaned.csv", "r")
line_input = fid.readlines()
fid.close()

# shuffle inputs and split into 70-30 (train/test respectively)
line_input = line_input[1:]
random.shuffle(line_input)

total_data_size = len(line_input)
data_size_70 = int(math.floor(0.7*(total_data_size)))  # need to be int for further use

# array[0] is the training set, array[1] is the test set
data_set = [line_input[:data_size_70], line_input[data_size_70+1:]]

file_names = ['train_a_data.csv', 'test_a_data.csv']

# write the data into seperate files
# ignore 1st line in ctg_data_cleaned as it contains headers
for idx in range(0, len(file_names)):
    current_file = open(file_names[idx], 'w')
    for item in data_set[idx]:
        current_file.write('%s' % item)
    current_file.close()

# split into k-fold
k_fold_number = 5  # use 5 for project
family = data_size_70 - (data_size_70 % k_fold_number)
for idx in range(0, k_fold_number):
    current_file = open('fold_' + str(idx)+'.csv', 'w')
    for item in range(int(idx/k_fold_number*family), int((idx+1)/k_fold_number*family)):
        current_file.write('%s' % data_set[0][item])
    if (idx == k_fold_number-1):  # add the orphaned inputs into the last fold
        for item in range(family, data_size_70-1):
            current_file.write('%s' % data_set[0][item])
    current_file.close()

# read input from csv file and convert to row and columns
fid = open("02\admission_predict.csv", "r")
line_input = fid.readlines()
fid.close()

# shuffle inputs and split into 70-30 (train/test respectively)
line_input = line_input[1:]
random.shuffle(line_input)

total_data_size = len(line_input)
data_size_70 = int(math.floor(0.7*(total_data_size)))  # need to be int for further use

# array[0] is the training set, array[1] is the test set
data_set = [line_input[:data_size_70], line_input[data_size_70+1:]]

file_names = ['train_b_data.csv', 'test_b_data.csv']

# write the data into seperate files
# ignore 1st line in ctg_data_cleaned as it contains headers
for idx in range(0, len(file_names)):
    current_file = open(file_names[idx], 'w')
    for item in data_set[idx]:
        current_file.write('%s' % item)
    current_file.close()
