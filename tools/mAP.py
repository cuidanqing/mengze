# -*- coding: utf-8 -*-
"""
根据比赛管方文档写的简单的mAP计算工具。
"""

import argparse
import numpy as np
import sys
import csv

def attr_p(probs, labels,ProbThreshold):
    """
    计算一个属性维度下的准确率, Stupid Version According to Official DOC
    """
    assert probs.shape == labels.shape

    BLOCK_COUNT = 0 # 不输出的个数
    PRED_COUNT = 0 # 预测输出的个数
    PRED_CORRECT_COUNT = 0 # 预测正确的个数
    GT_COUNT = probs.shape[0] # 该属性维度下所有相关数据的总条数
    for i, prob in enumerate(probs):
        index = np.argmax(prob)
        MaxAttrValueProb = prob[index]
        if MaxAttrValueProb < ProbThreshold:
            BLOCK_COUNT += 1
        else:
            if labels[i][index] == 'y':
                PRED_COUNT += 1
                PRED_CORRECT_COUNT += 1
            elif labels[i][index] == 'm':
                pass
            elif labels[i][index] == 'n':
                PRED_COUNT += 1
    P = float(PRED_CORRECT_COUNT) / PRED_COUNT
    return P

def gen_thresholds(probs):
    """
    返回某一属性维度下的所有阈值
    """
    BLOCK_COUNT = 0
    max_probs = np.max(probs, axis = 1)
    _threshold = sorted(set(max_probs))
    return _threshold

def attr_ap(probs, labels):
    _probs = []
    _labels = []
    for k in probs:
        _probs.append(probs[k])
        _labels.append(labels[k])
    _probs = np.array(_probs)
    _labels = np.array(_labels)
    
    _ap = 0.0
    thresholds = gen_thresholds(_probs)
    for threshold in thresholds:
        _p = attr_p(_probs, _labels, threshold)
        _ap += _p
    _ap /= len(thresholds)
    return _ap

def mAP(ground_truth, predicted):
    with open(ground_truth) as f:
        reader = csv.reader(f)
        ground_arr = np.array(list(reader))
    with open(predicted) as f:
        reader = csv.reader(f)
        predicted_arr = np.array(list(reader))

    assert ground_arr.shape == ground_arr.shape
    _map = 0.0
    attr_keys = set(ground_arr[:, 1])
    for attr_key in attr_keys:

        predicted_attr_arr = predicted_arr[predicted_arr[:, 1] == attr_key]
        predicted_prob_arr = np.array(map(lambda x: map(float, x.split(";")), predicted_attr_arr[:, 2]))

        ground_attr_arr = ground_arr[ground_arr[:,1] == attr_key]
        ground_label_arr = np.array(map(list, ground_attr_arr[:, 2]))

        assert predicted_prob_arr.shape == ground_label_arr.shape
        
        probs = dict(zip(predicted_attr_arr[:, 0], predicted_prob_arr))
        labels = dict(zip(ground_attr_arr[:, 0], ground_label_arr))

        _attr_ap = attr_ap(probs, labels)
        _map += _attr_ap * float(ground_attr_arr.shape[0]) / ground_arr.shape[0] # 求加权平均值

    return _map

def parse_args():
    parser = argparse.ArgumentParser(description = "map calculator")
    parser.add_argument("-g", "--ground", help="ground truth filename to be calculated", type=str, dest="ground")
    parser.add_argument("-p", "--predicted", help="predicted filename to be calculated", type=str, dest="predicted")

    return parser

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    ground_truth = args.ground
    predicted = args.predicted

    if not ground_truth or not predicted:
        print "[ERROR]: too few arguments."
        print ""
        parser.print_help()
        sys.exit()
    
    _map = mAP(ground_truth, predicted)
    print ">>> map = {0}".format(_map)
