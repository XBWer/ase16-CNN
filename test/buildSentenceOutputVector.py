# -*- coding: utf-8 -*-
import cPickle as pickle
import gensim
import glob
import os.path as op
from cs231n.classifiers.cnn_layer import *
import time
from path_config import index_npy, word2vec_model_path


def Index2sentenceVec(sentencesindex):
    sent_vector = np.array([])
    for index in sentencesindex:
        temp = word2vector_model[word2vector_model.index2word[index]]
        sent_vector = np.append(sent_vector, temp)
    return sent_vector


def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("../parameter/saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
    print st
    if st > 0:
        with open("../parameter/saved_params_%d.npy" % st, "r") as f:
            best_params = pickle.load(f)
            params = pickle.load(f)
            train_acc_history = pickle.load(f)
            val_acc_history = pickle.load(f)
            loss_history = pickle.load(f)
            epoch = pickle.load(f)
            state = pickle.load(f)
        # print params.keys()
        print 'Load paramenter%d' % st
        return st, best_params, params, train_acc_history, val_acc_history, loss_history, epoch, state
    else:
        return st, None, None, None, None, None, None, None


if __name__ == '__main__':
    print  'start training : ', time.strftime('%Y-%m-%d %H:%M:%S')

    word2vector_model = gensim.models.Word2Vec.load(word2vec_model_path)

    with open(index_npy, "r") as f:
        sentenceIndex = pickle.load(f)

    outputDim = 50
    WordVec_dim = 200
    reg_factor = 1e-6
    model = SentConvNet(
        WordVec_dim=WordVec_dim, output_dim=outputDim, weight_scale=1e-3, reg=reg_factor,
        dtype=np.float32)

    start_iter, bestparams_bk, oldparams_bk, train_acc_history_bk, val_acc_history_bk, loss_history_bk, epoch_bk, state = load_saved_params()
    model.params = oldparams_bk
    model_params = model.params

    sent_outputvector = []
    for i in range(len(sentenceIndex)):
        if i % 10000 == 0:
            print i, time.strftime('%Y-%m-%d %H:%M:%S')
        # transfer each word in sentence to index
        senttmp = sentenceIndex[i]
        # transfer each wordindex to model word vector : 200Dim
        sent_vector = Index2sentenceVec(senttmp)
        # input sentence vector to model and output 50Dim
        data = np.array(sent_vector).astype(np.float64).reshape(1, 1, 1, len(sent_vector))
        SentVector = model.outputvector(data, model_params)
        sent_outputvector.append(SentVector)

    sent_output_vectornpy = open("./sent_outputvec.npy", "w")
    pickle.dump(sent_outputvector, sent_output_vectornpy)
    f.close()

    print 'Finished : ', time.strftime('%Y-%m-%d %H:%M:%S')
