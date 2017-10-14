import os
import cv2
import numpy as np

# Get all files' path in training set
all_paths = {}
for (dirpath, dirnames, filenames) in os.walk("./test/"):
    class_name = dirpath.split('/')[-1]
    all_paths[class_name] = []
    for name in filenames:
        if name.split('.')[-1] != 'jpg':
            continue
        all_paths[class_name].append((dirpath, dirpath + '/' + name, name))
del all_paths['']

# Features extraction
for class_name, paths in all_paths.items():
    for p in paths:
        dir_path, file_path, file_name = p
        if file_name.split('.')[-1] != 'jpg':
            continue
        img = np.array(cv2.imread(file_path))
        features = []
        for i in range(int(len(img)/64)):
            for j in range(int(len(img[0])/64)):
                patch = np.array([p[j*64:(j+1)*64] for p in img[i*64:(i+1)*64]])
                
                color_histogram = np.zeros(24).astype(np.int)
                for p_i in range(len(patch)):
                    for p_j in range(len(patch[0])):
                        for color in range(3):
                            pixel = patch[p_i][p_j][color]
                            color_histogram[color*8 + int(pixel/32)] += 1
                features.append(color_histogram)
        features = np.array(features).astype(np.int)
        print("Writing " + dir_path + "/features_" + file_name.split('.')[0] + ".csv")
        np.savetxt(dir_path + "/features_" + file_name.split('.')[0] + ".csv", features, delimiter = ' ', fmt = '%s')
