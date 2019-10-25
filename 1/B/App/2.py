import matplotlib.pylab as plt
import numpy as np
import pandas


def main():
    # process data
    data = np.genfromtxt("../Data/train_data.csv", delimiter=",")
    corr_data = data[:, 1:9]

    # execute model
    test_dataframe = pandas.DataFrame(
        corr_data,
        columns=[
            "GRE Score",
            "TOEFL Score",
            "University Rating",
            "SOP",
            "LOR",
            "CGPA",
            "Research",
            "Chance of Admit",
        ],
    )
    plt.matshow(test_dataframe.corr())
    # use shape 1 for x-axis
    plt.xticks(
        range(test_dataframe.shape[1]), test_dataframe.columns, fontsize=7, rotation=90
    )
    plt.yticks(range(test_dataframe.shape[1]), test_dataframe.columns, fontsize=7)
    plt.colorbar()
    plt.legend()
    plt.savefig("../Out/2.png")


if __name__ == "__main__":
    main()
