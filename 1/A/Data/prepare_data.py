import math
import random

# import numpy as np

# * parameters
k_fold = 5
random.seed(291)

# * import/read raw dataset
input_filename = "ctg_data_cleaned.csv"
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

# # * k-fold split dataset
# even_split_number = dataset_size_70 - (dataset_size_70 % k_fold)
# per_even_split_entries = int(even_split_number / k_fold)
# k_datasets = []

# for i in range(0, even_split_number, per_even_split_entries):
#     k_datasets.append(dataset_split[0][i : (i + per_even_split_entries)])

# # add unappended entries (i.e. those that cannot be split evenly)
# # to final set
# k_datasets[k_fold - 1] = np.concatenate(
#     (k_datasets[k_fold - 1], dataset_split[0][even_split_number:])
# )

# # * export/write k-fold split dataset
# for i in range(k_fold):
#     with open("train_fold_" + str(i) + ".csv", "w") as f:
#         for entry in k_datasets[i]:
#             f.write("%s" % entry)
