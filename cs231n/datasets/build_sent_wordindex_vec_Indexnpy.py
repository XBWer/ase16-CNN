# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import gensim
import os

from path_config import medium_SentenceData_filepath, medium_index_npy

# sentence word->index in model
# 把每个sentence转换成word_index_vector
# generate Index.npy

modelpath = os.path.dirname(__file__) + '/model/model'
model = gensim.models.Word2Vec.load(modelpath)

WordVectors_len = model.vector_size

indexnpy = open(medium_index_npy, "w")
miss_word = 0
total_word = 0
line_num = 0
line_with_miss = 0
sent_vec_list = []
with open(medium_SentenceData_filepath, 'r') as f:
    for line in f:
        sent_split = line.decode('utf-8').strip().split(' ')
        sent_vec = []
        flag_miss = 0
        for sent in sent_split:
            total_word += 1
            if sent not in model:
                miss_word += 1
                flag_miss = 1
            else:
                sent_vec += list([model.vocab[sent].index])
        sent_vec_list.append(np.array(sent_vec).astype(np.int))
        if flag_miss == 1:
            line_with_miss += 1
        line_num += 1
print "total num of words:%d" % total_word
print "total num of missing words:%d" % miss_word
print "ratio of missing words:%f" % (float(miss_word) / float(total_word))
print "total num of sent:%d" % line_num
print "total num of sent with missing words:%d" % line_with_miss
print "ratio of missing sents:%f" % (float(line_with_miss) / float(line_num))
pickle.dump(sent_vec_list, indexnpy)
f.close()

# index.npy:
# sentence with each word index
