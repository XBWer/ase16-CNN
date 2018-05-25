from gensim.models.word2vec import Word2Vec, LineSentence
import time


print  'start time : ', time.strftime('%Y-%m-%d %H:%M:%S')
sentences = LineSentence('corpus.txt')

# size is the dimensionality of the feature vectors.
# window is the maximum distance between the current and predicted word within a sentence.
# min_count = ignore all words with total frequency lower than this.
# workers = use this many worker threads to train the model (=faster training with multicore machines).

model = Word2Vec(sentences, size=200, window=5, min_count=0, workers=4, iter=100)


model.save('model')
print  'end time : ', time.strftime('%Y-%m-%d %H:%M:%S')