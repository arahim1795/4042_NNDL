import math
import numpy as np
import random

# * parameters
k_fold = 5
random.seed(291)

# * import/read raw dataset
input_filename = "admission_predict.csv"
with open(input_filename, "r") as f:
    dataset = f.readlines()

# * cleanse dataset (i.e. remove header)
dataset = dataset[1:]

# * shuffle relavant inputs
random.shuffle(dataset)

# * split dataset
dataset_size = len(dataset)
dataset_size_70 = int(math.floor(0.7 * (dataset_size)))
dataset_split = [dataset[:dataset_size_70], dataset[dataset_size_70:]]

# * export/write split dataset
output_filenames = ["train", "test"]

for i in range(len(output_filenames)):
    with open(output_filenames[i] + "_data.csv", "w") as f:
        for entry in dataset_split[i]:
            f.write("%s" % entry)
