#-*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import csv
import random

class TianchiSingle(Dataset):
    """
    Dataset wrapper for one attribute dimension
    """
    def __init__(self, label_path, ):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

class TianchiMulti(Dataset):
    """
    Train
    """
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        with open(label_path) as f:
            reader = csv.reader(f)
            self.inputs = list(reader)

        random.shuffle(self.inputs)
        
        for line in self.inputs:
            line[2] = line[2].index('y')

    def __getitem__(self, index):
         img_path, attr_key, attr_value  = self.inputs[index]
         img_path = os.path.join(self.data_path, img_path)
         img = cv2.imread(img_path)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
         img = np.rollaxis(img, 2, 0)
         return img, attr_key, attr_value

    def __len__(self):
        return len(self.inputs)

class TianchiMultiInfer(Dataset):
    """
    Infer
    """
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        with open(label_path) as f:
            reader = csv.reader(f)
            self.inputs = list(reader)

    def __getitem__(self, index):
         raw_img_path, attr_key, _  = self.inputs[index]
         img_path = os.path.join(self.data_path, raw_img_path)
         img = cv2.imread(img_path)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
         img = np.rollaxis(img, 2, 0)
         return img, raw_img_path, attr_key

    def __len__(self):
        return len(self.inputs)


