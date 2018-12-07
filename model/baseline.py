# !/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import sys
import os
import pickle
from collections import defaultdict
from functools import reduce
from hold.ngram import Ngram


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.append(root_path)

class MaxProbModel:
    def __init__(self,n,training_path,test_path,predict_path):
        self.ngram = Ngram(n,training_path)
        self.get_vocabulary_probs(test_path)
        self.get_stats_probs()
        self.get_seg_result(test_path,predict_path)

    def get_seg_results(self,test_path,predict_path):
        self.seg_results = []
        with open(test_path,'r',encoding='utf-8') as f:
            for line in f:
                seg_result = self.segment(line)
                self.seg_results.append(seg_result)
            f.close()
        with open(predict_path,'w',encoding='utf-8') as f:
            for line in self.seg_results:
                f.write(line+'\n')
            f.close()

    def get_vocabulary_probs(self,test_path):
        with open(test_path,'r',encoding='utf-8') as f:
            char_set = set()
            for line in f:
                for char in line.strip():
                    char_set.add(char)
            f.close()
        char_list = list(char_set)
        vocabulary_counts = self.ngram.vocabulary_counts
        for char in char_list:
            vocabulary_counts[char] = vocabulary_counts.get(char,0)+1
        vocabulary_counts_keys,vocabulary_counts_values = zip(*list(vocabulary_counts.items()))
        vocabulary_counts_values_array = np.array(vocabulary_counts_values)
        vocabulary_counts_probs = tuple(vocabulary_counts_values_array/vocabulary_counts_values_array.sum())
        self.vocabulary_probs = dict(zip(vocabulary_counts_keys,vocabulary_counts_probs))

    def get_







    def segment(self,line):

        find_head
        dynamic
        get_result
        get_max_result



