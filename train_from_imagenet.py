# -*- coding: utf-8 -*-

"""
train ResNet-50 from image_net pretrained models
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time
import config

from tianchidataset import TianchiMulti
import models 

torch.cuda.set_device(2)
train_path = "/home/qyuan/tianchi/base" # path of training data
label_path = "/home/qyuan/tianchi/base/Annotations/label.csv" # path of training(base) label
log_save_path = "train_from_imagenet.log" # path to save training log
models_save_dir = "model_from_imagenet_pt" # path to save trained model

def train(net, data_loader, optimizer, criterion, num_epochs = 10):
    """
    train model
    :param net: train net
    :param data_loader: train data loader
    :param optimizer: optimizer
    :param num_epochs: epochs to train
    """
    
    with open(log_save_path, "w+") as f: pass # clear log
    if not os.path.exists(models_save_dir): os.makedirs(models_save_dir)
    
    time_start = time.time()

    def print_and_save_log(epoch, ite, loss):
        """
        print and save training log
        :param epoch: epoch
        :param ite: iteration
        :param loss: training loss
        """
        time_elapsed = time.time() - time_start
        cur_iter = epoch * len(data_loader) + ite
        if cur_iter == 0: return
        total_iter = num_epochs * len(data_loader)

        etc = time_elapsed * (float(total_iter) / cur_iter - 1)
        time_elapsed, etc = int(time_elapsed), int(etc)

        msg = "epoch: {}/{}, step: {}/{}, loss: {} | passed: {:.0f}h {:.0f}m {:.0f}s, etc: {:.0f}h {:.0f}m {:.0f}s".format(
                epoch, num_epochs, ite, len(data_loader), loss,
                time_elapsed / 3600,  time_elapsed % 3600 / 60, time_elapsed % 60,
                etc / 3600, etc % 3600 / 60, etc % 60
                )
        
        print msg 
        with open(log_save_path, "a") as f:
            f.write(msg + "\n")
    
    # train
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            
            imgs, attr_keys, attr_value = data
            attr_keys = np.array(attr_keys)
            
            imgs = Variable(imgs.float()).cuda()
            attr_value = Variable(attr_value).cuda()

            optimizer.zero_grad()
            attr_keys_set = set(attr_keys)
            
            loss = 0.0
            
            for key in attr_keys_set:
                index = np.where(attr_keys == key)[0]
                sub_imgs = imgs.index_select(0, torch.cuda.LongTensor(index))
                sub_labels = attr_value.index_select(0, torch.cuda.LongTensor(index))
                output = net(sub_imgs, key)
                loss += criterion(output, sub_labels)
            
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print_and_save_log(epoch, i, loss.data[0])

            if i % 1000 == 0: # save model snapshot
                torch.save(net.state_dict(), os.path.join(models_save_dir, net.__class__.__name__ + "_epoch_" + str(epoch) +  "_iter_" + str(i) + ".pth"))

if __name__ == "__main__":

    print ("loadding train data ...")
    dataset = TianchiMulti(train_path, label_path)
    data_loader = DataLoader(dataset, batch_size = 32)

    num_classes = config.num_classes
    
    criterion = nn.CrossEntropyLoss()

    net = models.resnet50_fashion(pretrained = False, num_classes = num_classes).cuda()
    net.load_state_dict(torch.load("./model_from_imagenet_pt/ResNetFashion_final.pth"))
    optimizer = torch.optim.Adam(params = net.parameters(), lr = 0.001)

    # train
    train(net, data_loader, optimizer, criterion, num_epochs = 100)
    torch.save(net.state_dict(), os.path.join(models_save_dir, net.__class__.__name__ + "_final" + ".pth"))
