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

class Grid:
    def __init__(self,word=None,prob=None,unk_num=None,seg=None):
        self.word = word
        self.prob = prob
        self.unk_num = unk_num
        self.seg = seg

    def reinit(self,word=None,prob=None,unk_num=None,seg=None):
        self.word = word
        self.prob = prob
        self.unk_num = unk_num
        self.seg = seg

class GridsTriangle:
    def __init__(self,sentence,two_gram):
        self.sentence = sentence
        self.sent_len = len(sentence)
        self.two_gram = two_gram
        self.vocabulary_prob =
        self.bulid_Grids()

    def init_tmp_grid(self,m,n,i):
        if

    def init_Grid(self,m,n):
        if m+n == self.sent_len-1:
            prob = self.two_gram.vocabulary
            self.Grids[m][n].reinit(self.sentence[m],)
        tmp_grids = [self.init_tmp_grid(m,n,i) for i in range(self.sent_len-m-n)]



    def build_Grids(self):
        self.Grids = []
        for i in range(self.sent_len):
            self.Grids.append([Grid() for j in range(self.sent_len-i)])
        for i in range(self.sent_len-1,-1,-1):
            for j in range(self.sent_len-1-i,-1,-1):
                self.init_Grid(i,j)

