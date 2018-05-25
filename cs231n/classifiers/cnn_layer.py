from cs231n.layers import *  # affine_backward,affine_forward,cosine_loss
from cs231n.fast_layers import *
from cs231n.layer_utils import *  # conv_relu_pool_forward,conv_relu_pool_backward
import numpy as np


class SentConvNet(object):
    """
      A two-channel-then-combined convolutional network with the following architecture:

      conv1.1 - relu1.1 -  max pool1.1 - affine - cosine similarity
      conv1.2 - relu1.2 -  max pool1.2 - affine - cosine similarity
      Extension : conv1.2 - relu1.2 -  max pool1.2 - affine - relu - affine - cosine similarity
      The network operates on minibatches of data that have shape (N, C, H, W)
      consisting of N images, each with height H and width W and with C input
      channels.
    """

    def __init__(self,
                 WordVec_dim=1, output_dim=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # 1,3,5,7,9-gram filters in the first layer
        self.filter_size11 = 1
        self.filter_size13 = 3
        self.filter_size15 = 5
        self.filter_size17 = 7
        self.filter_size19 = 9
        # Num of each i-gram filter is 128
        self.num_filters1 = 128

        self.WordVec_dim = WordVec_dim

        self.C1 = 1  # for text, only one channel

        # [num_filters1,1,1,1*200]
        self.params['W11'] = weight_scale * np.random.randn(self.num_filters1, self.C1, 1,
                                                            self.filter_size11 * WordVec_dim)
        # [64] all zero
        self.params['b11'] = np.zeros(self.num_filters1)  # bias_scale * np.random.randn(num_filters)
        # [1,1,1,3*200]
        self.params['W13'] = weight_scale * np.random.randn(self.num_filters1, self.C1, 1,
                                                            self.filter_size13 * WordVec_dim)
        self.params['b13'] = np.zeros(self.num_filters1)  # bias_scale * np.random.randn(num_filters)
        # [1,1,1,5*200]
        self.params['W15'] = weight_scale * np.random.randn(self.num_filters1, self.C1, 1,
                                                            self.filter_size15 * WordVec_dim)
        self.params['b15'] = np.zeros(self.num_filters1)  # bias_scale * np.random.randn(num_filters)
        # [1,1,1,7*200]
        self.params['W17'] = weight_scale * np.random.randn(self.num_filters1, self.C1, 1,
                                                            self.filter_size17 * WordVec_dim)
        self.params['b17'] = np.zeros(self.num_filters1)  # bias_scale * np.random.randn(num_filters)
        # [1,1,1,9*200]
        self.params['W19'] = weight_scale * np.random.randn(self.num_filters1, self.C1, 1,
                                                            self.filter_size19 * WordVec_dim)
        self.params['b19'] = np.zeros(self.num_filters1)  # bias_scale * np.random.randn(num_filters)
        # [64*5,50]
        self.params['Wout'] = weight_scale * np.random.randn(self.num_filters1 * 5, output_dim)
        self.params['bout'] = np.zeros(output_dim)  # bias_scale * np.random.randn(hidden_dim)

        # self.params['Wout'] = weight_scale * np.random.randn(self.num_filters2 * 5, output_dim)
        # self.params['bout'] = np.zeros(output_dim)  # bias_scale * np.random.randn(hidden_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    # X1:sent_vec1,sent_vec2,score
    def loss(self, X1, X2, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # zero_pad  : add 0 to left and right
        pad = 9 * self.WordVec_dim
        X1 = np.pad(X1, [(0, 0), (0, 0), (0, 0), (pad, pad)], 'constant')
        X2 = np.pad(X2, [(0, 0), (0, 0), (0, 0), (pad, pad)], 'constant')
        # print "X1"
        # print X1.shape

        W11, b11 = self.params['W11'], self.params['b11']
        W13, b13 = self.params['W13'], self.params['b13']
        W15, b15 = self.params['W15'], self.params['b15']
        W17, b17 = self.params['W17'], self.params['b17']
        W19, b19 = self.params['W19'], self.params['b19']
        # print "W11"
        # print W11.shape


        Wout, bout = self.params['Wout'], self.params['bout']
        C1 = self.C1
        H1, H2 = X1.shape[2], X2.shape[2]
        Width1, Width2 = X1.shape[3], X2.shape[3]
        reg = self.reg
        # pass conv_param to the forward pass for the convolutional layer
        # filter_size = W1.shape[2]
        conv_param1 = {'stride': self.WordVec_dim, 'pad': 0}

        # pass pool_param to the forward pass for the max-pooling layer


        pool_param111 = {'pool_height': H1, 'pool_width': (Width1 - W11.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param113 = {'pool_height': H1, 'pool_width': (Width1 - W13.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param115 = {'pool_height': H1, 'pool_width': (Width1 - W15.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param117 = {'pool_height': H1, 'pool_width': (Width1 - W17.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param119 = {'pool_height': H1, 'pool_width': (Width1 - W19.shape[3]) / self.WordVec_dim + 1, 'stride': 1}

        pool_param121 = {'pool_height': H2, 'pool_width': (Width2 - W11.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param123 = {'pool_height': H2, 'pool_width': (Width2 - W13.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param125 = {'pool_height': H2, 'pool_width': (Width2 - W15.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param127 = {'pool_height': H2, 'pool_width': (Width2 - W17.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param129 = {'pool_height': H2, 'pool_width': (Width2 - W19.shape[3]) / self.WordVec_dim + 1, 'stride': 1}

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        ################# Channel 1 - sentence ######################
        # layer | channel | gram
        # layer 1
        o111, cache111 = conv_relu_pool_forward(X1, W11, b11, conv_param1, pool_param111)
        o113, cache113 = conv_relu_pool_forward(X1, W13, b13, conv_param1, pool_param113)
        o115, cache115 = conv_relu_pool_forward(X1, W15, b15, conv_param1, pool_param115)
        o117, cache117 = conv_relu_pool_forward(X1, W17, b17, conv_param1, pool_param117)
        o119, cache119 = conv_relu_pool_forward(X1, W19, b19, conv_param1, pool_param119)
        # print "o111"
        # print o111.shape

        of1 = np.stack((o111, o113, o115, o117, o119), axis=3).reshape(
            np.append(o111.shape[0:3], o111.shape[3] * 5)).transpose(0, 3, 2, 1)
        # of1 = np.stack((o211, o213, o215, o217, o219), axis=3).reshape(np.append(o211.shape[0:3],o211.shape[3]*5)).transpose(0,3,2,1)
        # print "of1"
        # print of1.shape
        # output layer
        oout1, cacheout1 = affine_forward(of1, Wout, bout)
        # print oout1.shape

        # o31, cache31 = affine_sampling_forward(o21, W3, b3)

        ################# Channel 2 - sentence2 ######################
        # layer 1
        o121, cache121 = conv_relu_pool_forward(X2, W11, b11, conv_param1, pool_param121)
        o123, cache123 = conv_relu_pool_forward(X2, W13, b13, conv_param1, pool_param123)
        o125, cache125 = conv_relu_pool_forward(X2, W15, b15, conv_param1, pool_param125)
        o127, cache127 = conv_relu_pool_forward(X2, W17, b17, conv_param1, pool_param127)
        o129, cache129 = conv_relu_pool_forward(X2, W19, b19, conv_param1, pool_param129)

        of2 = np.stack((o121, o123, o125, o127, o129), axis=3).reshape(
            np.append(o121.shape[0:3], o121.shape[3] * 5)).transpose(0, 3, 2, 1)
        # of2 = np.stack((o221, o223, o225, o227, o229), axis=3).reshape(np.append(o221.shape[0:3],o221.shape[3]*5)).transpose(0,3,2,1)
        # output layer
        oout2, cacheout2 = affine_forward(of2, Wout, bout)
        # o32, cache32 = affine_sampling_forward(o21, W3, b3)
        # scores, cache3 = affine_forward(o2, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        similarity = None
        if y is None:
            oout1_norm = np.sqrt(np.sum(oout1 ** 2, axis=1, keepdims=True))
            oout2_norm = np.sqrt(np.sum(oout2 ** 2, axis=1, keepdims=True))
            prod = np.sum(oout1 * oout2, axis=1, keepdims=True)
            similarity = prod / (oout1_norm * oout2_norm)
            return similarity

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # oout1,oout2 shape : [1,50]
        # if np.sqrt(np.sum(oout1**2,axis=1,keepdims=True))==0 or  np.sqrt(np.sum(oout2**2,axis=1,keepdims=True))==0:
        #     print 'X1',X1
        #     print 'X2',X2

        # loss fucntion
        data_loss, doout1, doout2 = cosine_loss(oout1, oout2, y)
        # data_loss, doout1, doout2 = softmax_loss(oout1, oout2, y)
        # print "doout1"
        # print doout1.shape
        # print "doout2"
        # print doout2.shape

        # Compute the gradients using a backward pass
        ###
        # Channel 1
        dof1, dWout1, dbout1 = affine_backward(doout1, cacheout1)
        do111, do113, do115, do117, do119 = np.split(dof1.transpose(0, 3, 2, 1), 5, axis=3)
        # Layer 1
        dX11, dW111, db111 = conv_relu_pool_backward(do111, cache111)
        dX13, dW113, db113 = conv_relu_pool_backward(do113, cache113)
        dX15, dW115, db115 = conv_relu_pool_backward(do115, cache115)
        dX17, dW117, db117 = conv_relu_pool_backward(do117, cache117)
        dX19, dW119, db119 = conv_relu_pool_backward(do119, cache119)

        ###
        # Channel 2
        # Output layer to layer 2
        dof2, dWout2, dbout2 = affine_backward(doout2, cacheout2)
        do121, do123, do125, do127, do129 = np.split(dof2.transpose(0, 3, 2, 1), 5, axis=3)
        # Layer 1
        dX21, dW121, db121 = conv_relu_pool_backward(do121, cache121)
        dX23, dW123, db123 = conv_relu_pool_backward(do123, cache123)
        dX25, dW125, db125 = conv_relu_pool_backward(do125, cache125)
        dX27, dW127, db127 = conv_relu_pool_backward(do127, cache127)
        dX29, dW129, db129 = conv_relu_pool_backward(do129, cache129)

        ## dW1*
        dW11 = dW111 + dW121
        db11 = db111 + db121

        dW13 = dW113 + dW123
        db13 = db113 + db123

        dW15 = dW115 + dW125
        db15 = db115 + db125

        dW17 = dW117 + dW127
        db17 = db117 + db127

        dW19 = dW119 + dW129
        db19 = db119 + db129

        dWout = dWout1 + dWout2
        dbout = dbout1 + dbout2

        # Add regularization
        dW11 += reg * W11
        dW13 += reg * W13
        dW15 += reg * W15
        dW17 += reg * W17
        dW19 += reg * W19

        dWout += reg * Wout
        # reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W11,W13,W15,W17,W19, W21,W23,W25,W27,W29,  Wout])
        reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W11, W13, W15, W17, W19, Wout])

        loss = data_loss[0] + reg_loss
        # print str(data_loss[0]) + ',' + str(reg_loss)
        # print "dataloss", data_loss
        # print 'reg_loss', reg_loss
        grads = {'W11': dW11, 'W13': dW13, 'W15': dW15, 'W17': dW17, 'W19': dW19,
                 'b11': db11, 'b13': db13, 'b15': db15, 'b17': db17, 'b19': db19,
                 'Wout': dWout, 'bout': dbout}

        ##add W31_mask W32_mask
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def outputvector(self, X, params=None):
        if params != None:
            self.params = params

        pad = 9 * self.WordVec_dim
        X = np.pad(X, [(0, 0), (0, 0), (0, 0), (pad, pad)], 'constant')

        W11, b11 = self.params['W11'], self.params['b11']
        W13, b13 = self.params['W13'], self.params['b13']
        W15, b15 = self.params['W15'], self.params['b15']
        W17, b17 = self.params['W17'], self.params['b17']
        W19, b19 = self.params['W19'], self.params['b19']
        # print "W11"
        # print W11.shape


        Wout, bout = self.params['Wout'], self.params['bout']
        C1 = self.C1
        H = X.shape[2]
        Width = X.shape[3]
        reg = self.reg
        # pass conv_param to the forward pass for the convolutional layer
        # filter_size = W1.shape[2]
        conv_param1 = {'stride': self.WordVec_dim, 'pad': 0}
        conv_param2 = {'stride': 1, 'pad': 0}

        # pass pool_param to the forward pass for the max-pooling layer


        pool_param11 = {'pool_height': H, 'pool_width': (Width - W11.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param13 = {'pool_height': H, 'pool_width': (Width - W13.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param15 = {'pool_height': H, 'pool_width': (Width - W15.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param17 = {'pool_height': H, 'pool_width': (Width - W17.shape[3]) / self.WordVec_dim + 1, 'stride': 1}
        pool_param19 = {'pool_height': H, 'pool_width': (Width - W19.shape[3]) / self.WordVec_dim + 1, 'stride': 1}

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        ################# Channel 1 ######################
        # layer | channel | gram
        # layer 1
        o11, cache11 = conv_relu_pool_forward(X, W11, b11, conv_param1, pool_param11)
        o13, cache13 = conv_relu_pool_forward(X, W13, b13, conv_param1, pool_param13)
        o15, cache15 = conv_relu_pool_forward(X, W15, b15, conv_param1, pool_param15)
        o17, cache17 = conv_relu_pool_forward(X, W17, b17, conv_param1, pool_param17)
        o19, cache19 = conv_relu_pool_forward(X, W19, b19, conv_param1, pool_param19)
        # print "o111"
        # print o111.shape
        of = np.stack((o11, o13, o15, o17, o19), axis=3).reshape(
            np.append(o11.shape[0:3], o11.shape[3] * 5)).transpose(0, 3, 2, 1)
        # of1 = np.stack((o211, o213, o215, o217, o219), axis=3).reshape(np.append(o211.shape[0:3],o211.shape[3]*5)).transpose(0,3,2,1)
        # print "of1"
        # print of1.shape
        # output layer
        oout, cacheout = affine_forward(of, Wout, bout)
        # print oout1.shape

        vector = oout
        return vector

    def convert_to_vector(self, data, params=None):

        output_vector = []

        ### need to modify
        for input_vec in data['sent_vector']:
            ## extract the sentence vector#######################
            sent1_index = input_vec
            sent1_vector = np.array([])
            for index in sent1_index:
                if index != -1:
                    temp = data['word_vector'][data['word_vector'].index2word[index]]
                else:
                    temp = np.zeros(data['word_vector'].vector_size)
                sent1_vector = np.append(sent1_vector, temp)

            output_vec = self.outputvector(sent1_vector.reshape(1, 1, 1, len(sent1_vector)), params)
            output_vector.append(output_vec)
            ###########
        return output_vector
