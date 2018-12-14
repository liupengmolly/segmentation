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
import copy
import time

class GridsTriangle(object):
    def __init__(self,sentence,ngram):
        self.sentence = sentence
        self.sent_len = len(sentence)
        self.ngram = ngram

    def __del__(self):
        for i in range(self.sent_len-1,-1,-1):
            for j in range(self.sent_len-1-i,-1,-1):
                del self.Grids[i][j]

    def init_tmp_grid(self,m,n,i,vocabulary_probs,n_gram_probs):
        if i==0:
            word = self.sentence[m:self.sent_len-n]
            prob = vocabulary_probs[word] if word in vocabulary_probs else None
            return prob
        else:
            right_grid = self.Grids[m][n+i]
            below_grid = self.Grids[self.sent_len-n-i][n]
            prob = right_grid[2] + below_grid[2] - vocabulary_probs[below_grid[0]]

            if right_grid[1] in n_gram_probs:
                if below_grid[0] in n_gram_probs[right_grid[1]]:
                    interval_condition_prob = n_gram_probs[right_grid[1]][below_grid[0]]
                else:
                    # interval_condition_prob = n_gram_probs[right_grid[1]]['<unk>']
                    interval_condition_prob = vocabulary_probs[below_grid[0]]
            else:
                # interval_condition_prob = n_gram_probs['<unk>']
                interval_condition_prob = vocabulary_probs[below_grid[0]]
            prob = prob+interval_condition_prob
            return prob

    def init_Grid(self,m,n,vocabulary_probs,n_gram_probs):
        if m+n == self.sent_len-1:
            if self.sentence[m] not in vocabulary_probs:
                return ValueError("{} not in the vocabulary_probs".format(self.sentence[m]))
            prob = vocabulary_probs[self.sentence[m]]
            self.Grids[m][n] = (self.sentence[m],self.sentence[m],prob,[])
            return 0.0
        else:
            probs = []
            t1 = time.clock()
            for i in range(self.sent_len-m-n):
                tmp_prob = self.init_tmp_grid(m,n,i,vocabulary_probs,n_gram_probs)
                probs.append(tmp_prob)
            t1 = time.clock()-t1
            max_prob = None
            max_index = None
            # for i in range(len(probs)):
            #     #这里可以对相同的概率的切分方式进行选择，这里选择第一个出现
            #     if i!=0 and (tmp_max_prob is None or probs[i]>tmp_max_prob):
            #         tmp_max_prob = probs[i]
            #         tmp_max_index = i
            # right_grid = self.Grids[m][n+tmp_max_index]
            # below_grid = self.Grids[self.sent_len-n-tmp_max_index][n]
            # tmp_word = right_grid[0]
            # tmp_seg = right_grid[3] +[self.sent_len-n-tmp_max_index]+below_grid[3]
            # if probs[0] is not None:
            #     full_word = self.sentence[m:self.sent_len-n]
            #     self.Grids[m][n] = (full_word,full_word,probs[0],[])
            #     if tmp_word in self.ngram.vocabulary_counts:
            #         if len(tmp_seg)>=2:
            #             end = tmp_seg[1]
            #         else:
            #             end = self.sent_len-n
            #         if self.sentence[tmp_seg[0]:end] in self.ngram.vocabulary_counts and \
            #             self.ngram.vocabulary_counts[full_word] < self.ngram.vocabulary_counts[tmp_word]:
            #                 self.Grids[m][n] = (tmp_word,below_grid[1],tmp_max_prob,tmp_seg)
            # else:
            #     self.Grids[m][n] = (tmp_word,below_grid[1],tmp_max_prob,tmp_seg)
            for i in range(len(probs)):
                # 这里可以对相同的概率的切分方式进行选择，这里选择第一个出现
                if (probs[i] is not None) and (max_prob is None or probs[i]>max_prob):
                    max_prob = probs[i]
                    max_index = i
            if max_index==0:
                word = self.sentence[m:self.sent_len-n]
                self.Grids[m][n] = (word, word,max_prob, [])
            else:
                right_grid = self.Grids[m][n+max_index]
                below_grid = self.Grids[self.sent_len-n-max_index][n]
                self.Grids[m][n]=(right_grid[0],
                                  below_grid[1],
                                  max_prob,
                                  right_grid[3] +[self.sent_len-n-max_index]+below_grid[3])
            return t1

    def build_Grids(self,vocabulary_probs,n_gram_probs):
        t1,t2 = 0.0,0.0
        self.Grids = []
        for i in range(self.sent_len):
            self.Grids.append([() for j in range(self.sent_len-i)])
        for i in range(self.sent_len-1,-1,-1):
            for j in range(self.sent_len-1-i,-1,-1):
                s1 = self.init_Grid(i,j,vocabulary_probs,n_gram_probs)
                t1+=s1
        return t1

