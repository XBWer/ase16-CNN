# -*- coding: utf-8 -*-
import cPickle as pickle
import numpy as np
from gensim.models.word2vec import Word2Vec


def load_Pair(ROOT, filename):
    with open(ROOT + filename, 'r') as f:
        X1 = []
        X2 = []
        score = []
        degree = []
        groupId1 = []
        groupId2 = []
        # format
        # id1
        # id2
        # groupId1,groupId2
        i = 0
        for line in f:
            if i % 5 == 0:
                X1.append(int(line.strip()))
            elif i % 5 == 1:
                X2.append(int(line.strip()))
            elif i % 5 == 2:
                # score.append(line.strip().split())
                score = line.strip().split()
            elif i % 5 == 3:
                degreetmp = int(line.strip())
                if degreetmp == 1:
                    degreetmp = 0.2
                elif degreetmp == 2:
                    degreetmp = 0.4
                elif degreetmp == 3:
                    degreetmp = 0.6
                else:
                    degreetmp = 0.8
                degree.append(degreetmp)
                # degree = int(line.strip())
            else:
                groups = line.strip().split(',')
                groupId1.append(groups[0])
                groupId2.append(groups[1])
            i += 1
        # score = np.array(score).astype(np.float64)
        degree = np.array(degree).astype(np.float64)
        groupId1 = np.array(groupId1).astype(np.int)
        groupId2 = np.array(groupId2).astype(np.int)
        return X1, X2, degree, groupId1, groupId2


def load_train_eval_data(ROOT):
    # load model
    xbwmodel = Word2Vec.load(ROOT + '/model/model')
    xbwsize = xbwmodel.vector_size

    with open(ROOT + '/Index.npy', "r") as f:
        sentenceIndex = pickle.load(f)

    X1Train, X2Train, ScoreTrain, GroupId1Train, GroupId2Train = load_Pair(ROOT, '/trainingPair.txt')
    X1Eval, X2Eval, ScoreEval, GroupId1Eval, GroupId2Eval = load_Pair(ROOT, '/evalPair.txt')

    return X1Train, X2Train, ScoreTrain, GroupId1Train, GroupId2Train, X1Eval, X2Eval, ScoreEval, GroupId1Eval, GroupId2Eval, xbwsize, sentenceIndex, xbwmodel


def get_train_eval_data(num_training=None, num_validation=None):
    """
    Load the QA dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw data
    root = '../cs231n/datasets-LinkRecommendation'
    X1Train, X2Train, ScoreTrain, GroupId1Train, GroupId2Train, X1Eval, X2Eval, ScoreEval, GroupId1Eval, GroupId2Eval, xbwsize, sentenceIndex, xbwmodel = load_train_eval_data(
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
        GroupId1Train = [GroupId1Train[i] for i in mask]
        GroupId2Train = [GroupId2Train[i] for i in mask]

    if num_validation is not None:
        num_val = min(num_validation, len(ScoreEval))
        mask = np.random.choice(len(ScoreEval), num_val, replace=False)
        X1Eval = [X1Eval[i] for i in mask]
        X2Eval = [X2Eval[i] for i in mask]
        ScoreEval = ScoreEval[mask]
        GroupId1Eval = [GroupId1Eval[i] for i in mask]
        GroupId2Eval = [GroupId2Eval[i] for i in mask]

    # Package data into a dictionary
    return {
        'X1_train': X1Train, 'X2_train': X2Train, 'ScoreTr': ScoreTrain, 'y1_train': GroupId1Train,
        'y2_train': GroupId2Train,
        'X1_val': X1Eval, 'X2_val': X2Eval, 'ScoreEval': ScoreEval, 'y1_val': GroupId1Eval,
        'y2_val': GroupId2Eval,
        'vector_dim': xbwsize,
        'sent_vector': sentenceIndex,
        'word_vector': xbwmodel
    }  # for new sentence generate new sentence_vector
