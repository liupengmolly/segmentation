# !/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import numpy as np
import sys
import os
import time
import re
import copy
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
            count = 0
            for line in f:
                # count += 1
                # if count == 100:
                #     break
                new_line = re.sub('([,.!?，。！？])',r'\1<ss>',line.strip())
                new_lines = new_line.split('<ss>')
                seg_result = ''
                for l in new_lines:
                    if l == '':
                        continue
                    tmp_seg_result,i1 = self.segment(l)
                    t1+=i1
                    seg_result += ' '+tmp_seg_result
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
        vocabulary_counts = copy.copy(self.ngram.vocabulary_counts)
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
            # smooth_sub_prob = np.log((smooth_sub_v+1)/(smooth_sub_v.sum()+vocabulary_size))
            smooth_sub_prob = np.log((smooth_sub_v)/(smooth_sub_v.sum()))
            smooth_sub_dict = dict(zip(sub_k,tuple(smooth_sub_prob)))
            # smooth_sub_dict['<unk>'] = np.log(1/(smooth_sub_v.sum()+vocabulary_size))
            self.n_gram_probs[k] = smooth_sub_dict
        # self.n_gram_probs['<unk>'] = np.log(1/vocabulary_size)

    def segment(self,line):
        process_line = re.sub('[０-９]','0',line)
        process_line = re.sub('[0-9]','0',process_line)
        process_line = re.sub('[○|一|二|三|四|五|六|七|八|久|十]', '十', process_line)
        grids = GridsTriangle(process_line,self.ngram)
        t1 = grids.build_Grids(self.vocabulary_probs,self.n_gram_probs)
        seg = grids.Grids[0][0][3]
        seg.insert(0,0)
        seg.append(len(line))
        seg_words = []
        tmp_word = ''
        #处理未出现词的结合
        for i in range(len(seg)-1):
            process_seg_word = process_line[seg[i]:seg[i+1]]
            seg_word = line[seg[i]:seg[i+1]]
            if process_seg_word in self.ngram.vocabulary_counts:
                if tmp_word!='':
                    seg_words.append(tmp_word)
                    tmp_word = ''
                seg_words.append(seg_word)
            else:
                tmp_word+=seg_word
        if tmp_word!='':
            seg_words.append(tmp_word)
        line = ' '.join(seg_words)
        return line,t1

model = MaxProbModel(2,
                     '../icwb2-data/training/pku_training.utf8',
                     '../icwb2-data/testing/pku_test.utf8',
                     '../icwb2-data/predict/pku_predict')




