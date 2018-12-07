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

class Grid(object):
    def __init__(self,word=None,last_word=None,prob=None,seg=None):
        self.word = word
        self.last_word = last_word
        self.prob = prob
        self.seg = seg

    def reinit(self,word=None,last_word=None,prob=None,seg=None):
        self.word = word
        self.last_word = last_word
        self.prob = prob
        self.seg = seg

class GridsTriangle(object):
    def __init__(self,sentence):
        self.sentence = sentence
        self.sent_len = len(sentence)

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
            prob = right_grid.prob+below_grid.prob-vocabulary_probs[below_grid.word]

            if right_grid.last_word in n_gram_probs:
                if below_grid.word in n_gram_probs[right_grid.last_word]:
                    interval_condition_prob = n_gram_probs[right_grid.last_word][below_grid.word]
                else:
                    interval_condition_prob = n_gram_probs[right_grid.last_word]['<unk>']
            else:
                interval_condition_prob = n_gram_probs['<unk>']
            prob = prob+interval_condition_prob
            return prob

    def init_Grid(self,m,n,vocabulary_probs,n_gram_probs):
        if m+n == self.sent_len-1:
            if self.sentence[m] not in vocabulary_probs:
                return ValueError("{} not in the vocabulary_probs".format(self.sentence[m]))
            prob = vocabulary_probs[self.sentence[m]]
            self.Grids[m][n].reinit(self.sentence[m],self.sentence[m],prob,[])
            return 0.0
        else:
            probs = []
            t1 = time.clock()
            for i in range(self.sent_len-m-n):
                tmp_grid = self.init_tmp_grid(m,n,i,vocabulary_probs,n_gram_probs)
                probs.append(tmp_grid)
            t1 = time.clock()-t1
            max_prob = None
            max_index = None
            for i in range(len(probs)):
                #这里可以对相同的概率的切分方式进行选择，这里选择第一个出现
                if (probs[i] is not None) and (max_prob is None or probs[i]>max_prob):
                    max_prob = probs[i]
                    max_index = i
            if max_index==0:
                word = self.sentence[m:self.sent_len-n]
                self.Grids[m][n].reinit(word, word,max_prob, [])
            else:
                right_grid = self.Grids[m][n+max_index]
                below_grid = self.Grids[self.sent_len-n-max_index][n]
                self.Grids[m][n].reinit(right_grid.word,
                                        below_grid.last_word,
                                        max_prob,
                                        right_grid.seg+[self.sent_len-n-max_index]+below_grid.seg)
            return t1

    def build_Grids(self,vocabulary_probs,n_gram_probs):
        t1,t2 = 0.0,0.0
        self.Grids = []
        for i in range(self.sent_len):
            self.Grids.append([Grid() for j in range(self.sent_len-i)])
        for i in range(self.sent_len-1,-1,-1):
            for j in range(self.sent_len-1-i,-1,-1):
                s1 = self.init_Grid(i,j,vocabulary_probs,n_gram_probs)
                t1+=s1
        return t1

