# !/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import sys
import os
import time
from collections import defaultdict

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.append(root_path)

from hold.ngram import Ngram
from hold.data_structure import GridsTriangle


class MaxProbModel(object):
    def __init__(self,n,training_path,test_path,predict_path):
        self.ngram = Ngram(n,training_path)
        self.get_vocabulary_probs(test_path)
        self.get_laplace_stats_probs()
        self.get_seg_results(test_path,predict_path)


    def get_seg_results(self,test_path,predict_path):
        self.seg_results = []
        with open(test_path,'r',encoding='utf-8') as f:
            t1,t2 = 0.0,time.clock()
            for line in f:
                seg_result,i1 = self.segment(line.strip())
                t1+=i1
                self.seg_results.append(seg_result)
            f.close()
            print(t1,time.clock()-t2)
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
        vocabulary_counts_probs = tuple(np.log(vocabulary_counts_values_array/vocabulary_counts_values_array.sum()))
        self.vocabulary_probs = dict(zip(vocabulary_counts_keys,vocabulary_counts_probs))

    def get_laplace_stats_probs(self):
        """
        拉普拉斯平滑法，只记录训练集中出现的词组
        前缀出现、词尾没有出现的统一默认为1/(N+V)
        前缀没有出现的统一默认为1/V
        :return:
        """
        vocabulary_size = len(list(self.vocabulary_probs.keys()))
        n_gram_stats = self.ngram.n_gram_stats
        self.n_gram_probs = defaultdict(dict)
        for k,v in n_gram_stats.items():
            sub_k,sub_v = zip(*list(v.items()))
            smooth_sub_v = np.array(sub_v)
            smooth_sub_prob = np.log((smooth_sub_v+1)/(smooth_sub_v.sum()+vocabulary_size))
            smooth_sub_dict = dict(zip(sub_k,tuple(smooth_sub_prob)))
            smooth_sub_dict['<unk>'] = np.log(1/(smooth_sub_v.sum()+vocabulary_size))
            self.n_gram_probs[k] = smooth_sub_dict
        self.n_gram_probs['<unk>'] = np.log(1/vocabulary_size)

    def segment(self,line):
        grids = GridsTriangle(line)
        t1 = grids.build_Grids(self.vocabulary_probs,self.n_gram_probs)
        seg = grids.Grids[0][0].seg
        for i in range(len(seg)):
            line = line.replace(line[seg[i]+i:],' '+line[seg[i]+i:])
        return line,t1

model = MaxProbModel(2,
                     '../icwb2-data/training/msr_training.utf8',
                     '../icwb2-data/testing/msr_test.utf8',
                     '../icwb2-data/predict/msr_predict')




