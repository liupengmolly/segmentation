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

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.append(root_path)

class Ngram(object):
    def __init__(self,n,datapath):
        self.n_gram_stats = None
        self.vocabulary_counts = None
        self.sorted_vocabulary = None
        self.source = datapath.split('/')[-1].split('_')[0]
        if n<=1:
            return ValueError("The arguemnt n must be larger than 1,now,n is {}".format(n))
        if os.path.exists('../cache/{}_{}_gram_stats.pickle'.format(self.source,n)):
            stats_file = open('../cache/{}_{}_gram_stats.pickle'.format(self.source,n),'rb')
            self.n_gram_stats = pickle.load(stats_file)
            stats_file.close()
            vocabulary_counts_file = open('../cache/{}_{}_gram_vocabulary_counts.pickle'.format(self.source,n),'rb')
            self.vocabulary_counts = pickle.load(vocabulary_counts_file)
            vocabulary_counts_file.close()
            sortd_vocabulary_file = open('../cache/{}_{}_gram_sorted_vocabulary.pickle'.format(self.source,n),'rb')
            self.sorted_vocabulary = pickle.load(sortd_vocabulary_file)
            sortd_vocabulary_file.close()
        else:
            self.get_gram_stat(n,datapath)

    def get_gram_stat(self,n,datapath):
        self.n_gram_stats = defaultdict(dict)
        self.vocabulary_counts = dict()
        with open(datapath,'r',encoding='utf-8') as f:
            for line in f:
                pattern = re.compile('( )+')
                line = re.sub(pattern,' ',line.strip())
                words = line.split(' ')
                len_words = len(words)
                if len_words == n-1:
                    continue
                for i in range(len_words):
                    self.vocabulary_counts[words[i]] = self.vocabulary_counts.get(words[i],0)+1
                    if i>=n-1:
                        prefix_key = tuple([word for word in words[i-n+1:i]])
                        self.n_gram_stats[prefix_key][words[i]] = self.n_gram_stats[prefix_key].get(words[i],0)+1
            f.close()
        self.sorted_vocabulary = sorted(list(self.vocabulary_counts.keys()))
        stats_file = open('../cache/{}_{}_gram_stats.pickle'.format(self.source,n),'wb')
        pickle.dump(self.n_gram_stats,stats_file)
        stats_file.close()
        vocabulary_counts_file = open('../cache/{}_{}_gram_vocabulary_counts.pickle'.format(self.source,n),'wb')
        pickle.dump(self.vocabulary_counts,vocabulary_counts_file)
        vocabulary_counts_file.close()
        sorted_vocabulary_file = open('../cache/{}_{}_gram_sorted_vocabulary.pickle'.format(self.source,n),'wb')
        pickle.dump(self.sorted_vocabulary,sorted_vocabulary_file)
        sorted_vocabulary_file.close()


    def show(self,):
        if self.n_gram_stats is None:
            return ValueError("the statistics of the training data is not prepared")
        print(self.n_gram_stats)

    def get_max_suffixword_len(self):
        suffixlists = [list(item.keys()) for item in list(self.n_gram_stats.values())]
        suffixwords = []
        for suffixlist in suffixlists:
            suffixwords.extend(suffixlist)
        print(len(suffixwords))#867538
        long_suffixwords= [i for i in suffixwords if len(i)<=5]
        print(len(long_suffixwords))#849759
        return long_suffixwords

    def get_dict(self):
        with open('../icwb2-data/scripts/{}_dict'.format(self.source),'w',encoding='utf-8') as f:
            for line in self.sorted_vocabulary:
                f.write(line+'\n')
            f.close()



if __name__ == '__main__':
    two_gram = Ngram(2,'../icwb2-data/training/pku_training.utf8')
    print(len(two_gram.sorted_vocabulary))#88118
    two_gram.get_dict()
    # two_gram.get_max_suffixword_len()
