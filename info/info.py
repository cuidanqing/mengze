# -*- coding: utf-8 -*-
"""
Statistical information about the training(base) data set.
the information is saved into a json file named `train_info.json`
"""

import csv
import json
import numpy as np
from collections import Counter


def count(path):

    with open(path) as f:
        reader = csv.reader(f)
        arr = np.array(list(reader))

    attr_keys = set(arr[:, 1])
    info = {
            "sample_num": arr.shape[0],
            "attrs": {}
            }

    for attr_key in attr_keys:
        attr_arr = arr[arr[:, 1] == attr_key]
        label_arr = np.array(map(list, attr_arr[:, 2]))
        attr_count = {}
        for i in range(label_arr.shape[1]):
            counter = dict(Counter(label_arr[:,i]))
            for label in ["m", "y", "n"]:
                if not counter.has_key(label):
                    counter[label] = 0
            attr_count[i] = counter
            attr_count[i]["weight"] = float(counter["y"] + counter["m"]) / attr_arr.shape[0] # y + m

        atrr_info = {
                "name": attr_key,
                "attr_num": label_arr.shape[1],
                "count": attr_count,
                "weight": float(attr_arr.shape[0]) / arr.shape[0],
                "sample_num": attr_arr.shape[0]
                }
        info["attrs"][attr_key] = atrr_info

    return info

if __name__ == "__main__":
    # config to your own training(base) label label path
    train_label_path = "/home/qyuan/tianchi/base/Annotations/label.csv"
    info = count(train_label_path)
    with open("train_info.json", "w+") as f:
        json.dump(info, f)

