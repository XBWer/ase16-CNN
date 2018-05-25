import os

project_dir = ''

dataset_dir = project_dir + '/cs231n/datasets'

word2vec_model_path = dataset_dir + os.sep + 'model/model'

data_dir = dataset_dir + os.sep + 'data'
testPair_filepath = data_dir + os.sep + 'test_data.txt'
trainingPair_filepath = data_dir + os.sep + 'train_data.txt'
SentenceData_filepath = data_dir + os.sep + 'sentence.txt'
index_npy = data_dir + os.sep + 'Index.npy'
