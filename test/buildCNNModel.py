# As usual, a bit of setup
# encoding=utf-8
from cs231n.classifiers.cnn_layer import *
from cs231n.data_utils import get_train_eval_data
from cs231n.fast_layers import *
from cs231n.solver_multi import solver_multi
import cPickle as pickle
import time

ISOTIMEFORMAT = '% Y - % m - % d % X'

small_data = get_train_eval_data()

WordVec_dim = small_data['vector_dim']
# X1/2 has the format of [[data1],[data2],..[dataN]] ---- data-i is variable-length list, N is the number of sample
# Y has the format of np.array(1,size=N) ---- N is the number of sample
learning_rate = 1e-6
update_rule = 'adam'
reg_factor = 1e-6
outputDim = 50

# init model
# trick : weight_scale = learning_rate * 1000
# WordVec_dim = 200 (input_vec_dim)
# output_dim = 50 (output_vec_dim)

model = SentConvNet(
    WordVec_dim=WordVec_dim, output_dim=outputDim, weight_scale=1e-3, reg=reg_factor,
    dtype=np.float64)
solver = solver_multi(model, small_data,
                      num_epochs=10000, batch_size=100,
                      update_rule=update_rule,
                      optim_config={
                          'learning_rate': learning_rate,
                      },
                      verbose=True, print_every=1)

# data

print 'start training : ', time.strftime('%Y-%m-%d %H:%M:%S')
solver.train()
print 'training finished : ', time.strftime('%Y-%m-%d %H:%M:%S')
sent_output_vec = model.convert_to_vector(small_data)

# sentence vector 50D
with open("./outputs/sent_output_vec.npy", "w") as f:
    pickle.dump(sent_output_vec, f)
