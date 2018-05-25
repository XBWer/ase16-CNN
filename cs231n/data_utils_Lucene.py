# -*- coding: utf-8 -*-
import cPickle as pickle
import numpy as np
from gensim.models.word2vec import Word2Vec


def load_Pair(ROOT, filename):
    with open(ROOT + filename, 'r') as f:
        X1 = []
        X2 = []
        score = []
        i = 0
        for line in f:
            if i % 3 == 0:
                X1.append(int(line.strip()))
            elif i % 3 == 1:
                X2.append(int(line.strip()))
            else:
                score.append(line.strip().split())
            i += 1
        score = np.array(score).astype(np.float64)
        print score.shape
        return X1, X2, score


def load_train_eval_data(ROOT):
    # load model
    xbwmodel = Word2Vec.load(ROOT + '/model/model')
    xbwsize = xbwmodel.vector_size

    with open(ROOT + '/Index.npy', "r") as f:
        sentenceIndex = pickle.load(f)

    X1Train, X2Train, ScoreTrain = load_Pair(ROOT, '/trainingPair.txt')
    X1Eval, X2Eval, ScoreEval = load_Pair(ROOT, '/evalPair.txt')

    return X1Train, X2Train, ScoreTrain, X1Eval, X2Eval, ScoreEval, xbwsize, sentenceIndex, xbwmodel


def get_train_eval_data(num_training=None, num_validation=None):
    """
    Load the QA dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw data
    root = '../cs231n/datasets-LinkRecommendation'
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

    # Package data into a dictionary
    return {
        'X1_train': X1Train, 'X2_train': X2Train, 'ScoreTr': ScoreTrain,
        'X1_val': X1Eval, 'X2_val': X2Eval, 'ScoreEval': ScoreEval,
        'vector_dim': xbwsize,
        'sent_vector': sentenceIndex,
        'word_vector': xbwmodel
    }  # for new sentence generate new sentence_vector
