# -*- coding: utf-8 -*-
"""
generate inference result
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from tianchidataset import TianchiMultiInfer
import models
import csv
import config

torch.cuda.set_device(0)

num_classes = config.num_classes

def _infer(net, data_loader):    
    """
    generate infer result
    """
    result = []
    m = nn.Softmax()
    for i, data in enumerate(data_loader):
        if i % 100 == 0: print "inference {:.4f}%".format(float(i)/len(data_loader))
        imgs, imgs_path, attr_keys = data
        imgs_path = np.array(imgs_path)
        attr_keys = np.array(attr_keys) 
        imgs = Variable(imgs.float()).cuda()

        attr_keys_set = set(attr_keys)
        tmp = {}
        for key in attr_keys_set:
            index = np.where(attr_keys == key)[0]
            sub_imgs = imgs.index_select(0, torch.cuda.LongTensor(index))
            sub_imgs_path = imgs_path[index]
            sub_attr_keys = attr_keys[index]
            output = net(sub_imgs, key)
            output = m(output)
            output = output.data.cpu().numpy()
            for i in range(len(sub_imgs_path)):
                prob = ";".join(map(lambda x: "{:.4f}".format(x), list(output[i])))
                tmp[sub_imgs_path[i]] = [sub_imgs_path[i], sub_attr_keys[i], prob]
        for key in imgs_path:
            result.append(tmp[key])

    return result

def inference(model_path, train_path, label_path, infer_path):
    
    dataset = TianchiMultiInfer(train_path, label_path)
    data_loader = DataLoader(dataset, batch_size = 32)

    net = models.resnet50_fashion(pretrained = False, num_classes = num_classes).cuda()
    net.load_state_dict(torch.load(model_path))
    
    result = _infer(net, data_loader)
    with open(infer_path, "w+") as f:
        writer = csv.writer(f)
        for line in result:
            writer.writerow(line)

if __name__ == "__main__":
   
    # config to your own path
    model_path = "./model_pt/ResNetFashion_epoch_34_iter_2000pth" # model used to inference
    train_path = "/home/qyuan/tianchi/base" # path of training(base) data
    label_path = "/home/qyuan/tianchi/base/Annotations/label.csv" # path of training(data) label
    infer_path = "/home/qyuan/tianchi/base/Annotations/label_infer.csv" # path to save infer result
    inference(model_path, train_path, label_path, infer_path)

    """
    rank_path = "/home/qyuan/tianchi/rank"
    label_path = "/home/qyuan/tianchi/rank/Tests/question.csv"
    infer_path = "/home/qyuan/tianchi/rank/Tests/question_infer.csv"
    inference(model_path, rank_path, label_path, infer_path)
    """
