import os
import csv
import numpy as np

k = 64


def get_mu_dict(mu_matrix, all_features):
    mu = {key: [] for key in range(len(mu_matrix))}
    for x in all_features:
        dist = [np.linalg.norm(x-u) for u in mu_matrix]
        min_dist_index = dist.index(min(dist))
        mu[min_dist_index].append(x)
    return mu

def get_mu_matrix(mu):
    mu_matrix = []
    for k, v in mu.items():
        mean_feature = np.mean(v, axis = 0)
        mu_matrix.append(mean_feature)
    return mu_matrix

# Get all files' path in training set
all_paths = {}
for (dirpath, dirnames, filenames) in os.walk("./train/"):
    class_name = dirpath.split('/')[-1]
    all_paths[class_name] = []
    for name in filenames:
        if name.split('.')[-1] != 'csv':
            continue
        all_paths[class_name].append((dirpath, dirpath + '/' + name, name))
del all_paths['']

# Features extraction
classes = {}
for class_name, paths in all_paths.items():
    all_features = []
    for (dirpath, path, file_name) in paths:
        result = list(csv.reader(open(path, "r"), delimiter=" "))
        all_features += result
    all_features = np.array(all_features).astype(np.int)
    
    mu_matrix = all_features[np.random.randint(all_features.shape[0], size=k), :]
    mu = get_mu_dict(mu_matrix, all_features)

    for i in range(1, 21):
        print("Iteration " + str(i))
        mu_matrix = get_mu_matrix(mu)
        mu = get_mu_dict(mu_matrix, all_features)

    print(mu_matrix)

    exit()
