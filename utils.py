import numpy as np
import csv
import matplotlib.pyplot as plt


def load_csv_class(path):
    with open(path) as file:
        return np.array([line.strip().split(' ') for line in file]).astype(np.float)

def generate_summary_class(data, ratio):
    total_entries = len(data)
    r = int(ratio * total_entries)
    return data[:r], data[r:]

def calculate_mean(data):
    return np.mean(data, 0)

def calculate_cov(data):
    return np.cov(data)

def gaussian_probability(x, mean, cov):
    n = len(x)
    return (1 / (pow(2 * np.pi, n/2) * pow(np.linalg.det(cov), 0.5))) * np.exp((-0.5)*(np.dot(np.dot(np.subtract(x, mean).T, np.linalg.inv(cov)), np.subtract(x, mean))))


if __name__ == '__main__':
    ratio = 0.75
    dirname = "RD_group16"
    files = ['class1.txt', 'class2.txt', 'class3.txt']

    dataset = []
    for file_name in files:
        dataset.append(load_csv_class('./' + dirname + '/' + file_name))

    for data in dataset:
        training_set, test_set = generate_summary_class(data, ratio)
        mean, cov = calculate_mean(training_set), calculate_cov(training_set.T)
        for x in training_set:
            p = gaussian_probability(x.T, mean, cov)
            print(p)

