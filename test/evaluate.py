# -*- coding: utf-8 -*-
import cPickle as pickle
from cs231n.classifiers.cnn_layer import *
from gensim.models.word2vec import Word2Vec
import time


def calc_x1_x2_similarity(index1, index2):
    idx1 = sent_outputvec[index1]
    idx2 = sent_outputvec[index2]
    # ocos1_norm = np.sqrt(np.sum(idx1 ** 2))
    # ocos2_norm = np.sqrt(np.sum(idx2 ** 2))
    # prod = np.sum(idx1 * idx2)
    # dist = 1 - prod / (ocos1_norm * ocos2_norm)  # calculate cosine distance instead of similarity

    x1_norm = np.sqrt(np.sum(idx1 ** 2, axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(idx2 ** 2, axis=1, keepdims=True))
    prod = np.sum(idx1 * idx2, axis=1, keepdims=True)
    cosine_sim = prod / (x1_norm * x2_norm)
    # print "cosine_sim:"
    # print cosine_sim
    # loss = np.sum(0.5 * (y.reshape(y.shape[0], 1) - cosine_sim) ** 2, axis=0) / y.shape[0]
    return cosine_sim[0][0]


##################### test sent
def load_testData():
    # format
    # "ID"
    # id1
    # id2
    # type  == > score
    # duplicate : 7/8
    # direct : 5/8
    # indirect : 3/8
    # isolated : 1/8
    with open('../cs231n/datasets/data/test_data.txt', 'r') as f:
        pair = []
        linenum = 0
        for line in f:
            if line.strip() == '\"ID\"':
                continue
            if linenum % 3 == 0:
                id1 = int(line.strip())
            elif linenum % 3 == 1:
                id2 = int(line.strip())
            elif linenum % 3 == 2:
                type = line.strip()
                # 1 : duplicate, 2: direct, 3: indirect, 4: isolated
                pair.append((id1, id2, type))
            linenum += 1
    # return
    return pair


def Index2sentenceVec(sentencesindex):
    sent_vector = np.array([])
    for index in sentencesindex:
        temp = word2vector_model[word2vector_model.index2word[index]]
        sent_vector = np.append(sent_vector, temp)
    return sent_vector


if __name__ == '__main__':
    print  'start computing : ', time.strftime('%Y-%m-%d %H:%M:%S')
    print 'run buildSentenceOutputVector before run this code!!!'

    word2vector_model = Word2Vec.load('../cs231n/datasets/model/model')

    with open('./sent_outputvec.npy', "r") as f:
        sent_outputvec = pickle.load(f)

    testData = load_testData()

    # firstline
    writeStr = 'id1,id2,label,predict\n'

    # 1: duplicate, 2: direct, 3: indirect, 4: isolated
    target = [7 / 8.0, 5 / 8.0, 3 / 8.0, 1 / 8.0]
    testcnt = 0
    labels = []
    predicted = []
    for (id1, id2, label) in testData:
        pred = calc_x1_x2_similarity(id1, id2)
        # find corresponding range
        pred_list = [abs(target[i] - pred) for i in range(len(target))]
        idx = pred_list.index(min(pred_list))

        # 1: duplicate, 2: direct, 3: indirect, 4: isolated
        pred_class = str(idx + 1)
        predicted.append(pred_class)
        labels.append(label)

        writeStr += (
            str(testcnt) + ',' + str(id1) + ',' + str(id2) + ',' + str(label) + ',' + str(pred_class) + "\n")
        testcnt += 1

    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report

    print("f1_score: {}".format(f1_score(labels, predicted, average="micro")))
    print(classification_report(labels, predicted, target_names=["1", "2", "3", "4"]))

    # writing result
    f = open("./result.csv", "w")
    f.write(writeStr)

    print  'Finished : ', time.strftime('%Y-%m-%d %H:%M:%S')
