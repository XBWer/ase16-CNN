# -*- coding: utf-8 -*-
import cPickle as pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from path_config import medium_trainingPair_filepath, medium_testPair_filepath,medium_index_npy,word2vec_model_path


def load_Pair(filepath):
    with open(filepath, 'r') as f:
        X1 = []
        X2 = []
        similarity = []
        # format
        # "ID"
        # id1
        # id2
        # type  == > score
        # duplicate : 7/8
        # direct : 5/8
        # indirect : 3/8
        # isolated : 1/8
        i = 0
        for line in f:
            if line.strip() == '\"ID\"':
                continue
            if i % 3 == 0:
                X1.append(int(line.strip()))
            elif i % 3 == 1:
                X2.append(int(line.strip()))
            elif i % 3 == 2:
                type = int(line.strip())
                # duplicate
                if type == 1:
                    sim = 7 / 8.0
                # direct
                elif type == 2:
                    sim = 5 / 8.0
                # indirect
                elif type == 3:
                    sim = 3 / 8.0
                # isolated
                else:
                    sim = 1 / 8.0
                similarity.append(sim)
            i += 1
        similarity = np.array(similarity).astype(np.float64)
        return X1, X2, similarity


def load_train_eval_data(ROOT):
    # load model
    w2vmodel = Word2Vec.load(word2vec_model_path)
    w2vsize = w2vmodel.vector_size

    with open(medium_index_npy, "r") as f:
        sentenceIndex = pickle.load(f)

    X1Train, X2Train, ScoreTrain = load_Pair(medium_trainingPair_filepath)
    X1Eval, X2Eval, ScoreEval = load_Pair(medium_testPair_filepath)

    return X1Train, X2Train, ScoreTrain, X1Eval, X2Eval, ScoreEval, w2vsize, sentenceIndex, w2vmodel


def get_train_eval_data(num_training=None, num_validation=None):
    """
    Load the QA dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw data
    root = '../cs231n/datasets'
    X1Train, X2Train, ScoreTrain, X1Eval, X2Eval, ScoreEval, xbwsize, sentenceIndex, xbwmodel = load_train_eval_data(
        root)

    print "length of Train data:%d" % len(ScoreTrain)
    print "length of Val data:%d" % len(ScoreEval)

    # Subsample the data
    if num_training is not None:
        num_train = min(num_training, len(ScoreTrain))
        mask = np.random.choice(len(ScoreTrain), num_train, replace=False)
        X1Train = [X1Train[i] for i in mask]
        X2Train = [X2Train[i] for i in mask]
        ScoreTrain = ScoreTrain[mask]

    if num_validation is not None:
        num_val = min(num_validation, len(ScoreEval))
        mask = np.random.choice(len(ScoreEval), num_val, replace=False)
        X1Eval = [X1Eval[i] for i in mask]
        X2Eval = [X2Eval[i] for i in mask]
        ScoreEval = ScoreEval[mask]
    print len(sentenceIndex)
    # Package data into a dictionary
    return {
        'X1_train': X1Train, 'X2_train': X2Train, 'ScoreTr': ScoreTrain,
        'X1_val': X1Eval, 'X2_val': X2Eval, 'ScoreEval': ScoreEval,
        'vector_dim': xbwsize,
        'sent_vector': sentenceIndex,
        'word_vector': xbwmodel
    }  # for new sentence generate new sentence_vector
