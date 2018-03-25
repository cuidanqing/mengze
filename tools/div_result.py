# -*- coding: utf-8 -*-

"""
generate random data to test mAP.py
"""

import random
import numpy as np
import csv

def rand_dist(label, _max = False):
    """
    if param `_max` is set, `y` in `label` map the highest probobility.
    """
    rand_data = np.array([random.random() for _ in range(len(label))])
    if _max:
        rand_data[label.index('y')], rand_data[label.index('n')] = rand_data.max(), rand_data.min()
    rand_data = rand_data / np.sum(rand_data)
    return list(rand_data)

if __name__ == "__main__":
    with open("./label.csv") as f:
        reader = csv.reader(f)
        lines = list(reader)

    # genrate normal
    for i, line in enumerate(lines):
        if i % 1000 == 0: print i
        rand_data = rand_dist(line[2])
        line[2] = ";".join(map(lambda x: "{:.4}".format(x), rand_data))

    with open("div_predicted.csv", "w") as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)

    # generate max
    for i, line in enumerate(lines):
        if i % 1000 == 0: print i
        rand_data = rand_dist(line[2])
        line[2] = ";".join(map(lambda x: "{:.4}".format(x), rand_data))

    with open("div_predicted_max.csv", "w") as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)
