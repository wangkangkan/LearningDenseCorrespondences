from . import rescale

import tensorflow as tf
import scipy.sparse
import numpy as np
import os

class DecodeModel(object):
    """
    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        nz: Size of latent variable.
        F_0: Number of graph input features.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    U: Upsampling matrix

    config: the configture for gpu
    
    Directories:
        dir_name: Directory for loading trained model.
    
    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
       
    """
    #which_loss l1 or l2 defaut=l1
    def __init__(self, L, D, U, F, K, p, nz, which_loss='l1', F_0=1, filter='chebyshev5', brelu='b1relu',
                pool='poolwT',unpool='poolwT', regularization=0, dropout=0, batch_size=100,
                dir_name=' '):
        # super(DecodeModel, self).__init__()
        
        # Keep the useful Laplacians only. May be zero.
        self.M_0 = L[0].shape[0]
        # Store attributes and bind operations.
        self.L, self.D, self.U, self.F, self.K, self.p, self.nz, self.F_0 = L, D, U, F, K, p, nz, F_0
        self.which_loss = which_loss
        self.regularization, self.dropout = regularization, dropout
        self.batch_size = batch_size
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.unpool = getattr(self, unpool)

        self.regularizers = []

        self.dir_name = dir_name
        #self.loss_weights = weight_tensor
        # Build the computational graph.
        #self.build_graph(self.M_0, self.F_0)
        # self.sess = None
        
    def build_graph(self, M_0, F_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.batch_size = 100
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_z = tf.placeholder(tf.float32, (self.batch_size, self.nz ), 'z')
           
            self.op_decoder = self._decode(self.ph_z, reuse=True)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=200)

    



    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = rescale.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x) # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2,0,1]) # N x Mp x Fin

        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _encode(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('encoder', reuse = reuse):
            N, Min, Fin = x.get_shape()
            print('encode:\n', x.shape)
            for i in range(len(self.F)):
                print(i)
                with tf.variable_scope('conv{}'.format(i + 1)):
                    with tf.name_scope('filter'):
                        print(x)
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                        print(x)
                        print('filter:', x.shape)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                        print('brelu:', x.shape)
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.D[i])
                        print('pool:', x.shape)
        return x
        
    def _encode_single(self, x, i, reuse=tf.AUTO_REUSE):
        var_name = 'encoder_singel_%d' % (i+1)
        with tf.variable_scope(var_name, reuse = reuse):
            N, Min, Fin = x.get_shape()
            print('encode:\n', x.shape)
        # for i in range(len(self.F)):
            print(i)
            with tf.variable_scope('conv{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    print(x)
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                    print(x)
                    print('filter:', x.shape)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                    print('brelu:', x.shape)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.D[i])
                    print('pool:', x.shape)


            # Fully connected hidden layers.
            # N, M, F = x.get_shape()
            # x = tf.reshape(x, [int(N), int(self.p[-1] * self.F[-1])])  # N x MF
            # print('reshape:', x.shape)
            # if self.nz:
            #     with tf.variable_scope('fc'):
            #         t = int(self.nz)
            #         x = self.fc(x, 2 * t)  # N x M0
            #         print('fc:', x.shape)
            #         mu = x[:, :t]  # N x M0
            #         self.mu = mu
            #         print('mu :', mu.shape)
            #         sigma = 1e-6 + tf.nn.softplus(x[:, t:])  # N x M0
            #         self.sigma = sigma
            #         print('sigma :', sigma.shape)
            #         x = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)  # N x M0
            #         print('Guass :', x.shape)
        return x#, mu, sigma
    
    def _encode_fc(self, x, nz, scope='encode_fc', reuse=False): 
        with tf.variable_scope(scope, reuse=reuse):
            # N, Min, Fin = x.get_shape()
            # print('encode:\n',x.shape)
            # for i in range(len(self.F) // 2):
            #     with tf.variable_scope('conv{}'.format(i+1)):
            #         with tf.name_scope('filter'):
            #             x = self.filter(x, self.L[i], self.F[i], self.K[i])
            #             # print(self.L[i], self.F[i], self.K[i])
            #             print('filter:',x.shape)
            #         with tf.name_scope('bias_relu'):
            #             x = self.brelu(x)
            #             print('brelu:',x.shape)
            #         with tf.name_scope('pooling'):
            #             x = self.pool(x, self.D[i])
            #             print('pool:',x.shape)
            #             print(i)
        
            # Fully connected hidden layers.
            N, M, F = x.get_shape()
            x = tf.reshape(x, [int(N), -1])  # N x MF
            print('reshape:',x.shape)
            if self.nz:
                with tf.variable_scope('fc'):
                    x = self.fc(x, int(nz))    # N x M0
                    # x = self.fc(x, int(self.nz[0]))    # N x M0
                    print('fc:',x.shape)
        return x
    def decode_fc(self, x, nz, scope='decode_fc', reuse=False): 
        with tf.variable_scope(scope, reuse=reuse):
            # N, Min, Fin = x.get_shape()
            # print('encode:\n',x.shape)
            # for i in range(len(self.F) // 2):
            #     with tf.variable_scope('conv{}'.format(i+1)):
            #         with tf.name_scope('filter'):
            #             x = self.filter(x, self.L[i], self.F[i], self.K[i])
            #             # print(self.L[i], self.F[i], self.K[i])
            #             print('filter:',x.shape)
            #         with tf.name_scope('bias_relu'):
            #             x = self.brelu(x)
            #             print('brelu:',x.shape)
            #         with tf.name_scope('pooling'):
            #             x = self.pool(x, self.D[i])
            #             print('pool:',x.shape)
            #             print(i)
        
            # Fully connected hidden layers.
            N = x.get_shape()[0]
            # N, M, F = x.get_shape()
            x = tf.reshape(x, [int(N), -1])  # N x MF
            print('reshape:',x.shape)
            # if self.nz:
            # for I in []

            with tf.variable_scope('fc'):
                x = self.fc(x, int(nz))    # N x M0
                # x = self.fc(x, int(self.nz[0]))    # N x M0
                print('fc:',x.shape)
            x = tf.reshape(x, [int(N), int(self.p[0]), int(self.F_0)])
            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                print('refilter:',x.shape)
        return x


    def _encode_sample_1(self, x, reuse=False):
        with tf.variable_scope('encode_sample', reuse=reuse):
            N, Min, Fin = x.get_shape()
            print('encode:\n',x.shape)
            for i in range(len(self.F) // 2):
                with tf.variable_scope('conv{}'.format(i+1)):
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                        # print(self.L[i], self.F[i], self.K[i])
                        print('filter:',x.shape)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                        print('brelu:',x.shape)
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.D[i])
                        print('pool:',x.shape)
                        print(i)
        
            # Fully connected hidden layers.
            #N, M, F = x.get_shape()
            # x = tf.reshape(x, [int(N), int(self.p[1]*self.F[1])])  # N x MF
            # print('reshape:',x.shape)
            # if self.nz:
            #     with tf.variable_scope('fc'):
            #         x = self.fc(x, int(self.nz[0]))    # N x M0
            #         print('fc:',x.shape)
        return x

    def _encode_sample_2(self, x, reuse=False):
        with tf.variable_scope('encode_sample', reuse=reuse):
            N, Min, Fin = x.get_shape()
            print('encode:\n',x.shape)
            for i in range(len(self.F) // 2, len(self.F)):
                with tf.variable_scope('conv{}'.format(i+1)):
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                        # print(self.L[i], self.F[i], self.K[i])
                        print('filter:',x.shape)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                        print('brelu:',x.shape)
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.D[i])
                        print('pool:',x.shape)
                        print(i)
        
            # Fully connected hidden layers.
            #N, M, F = x.get_shape()
            # x = tf.reshape(x, [int(N), int(self.p[-1]*self.F[-1])])  # N x MF
            # print('reshape:',x.shape)
            # if self.nz:
            #     with tf.variable_scope('fc'):
            #         x = self.fc(x, int(self.nz))    # N x M0
            #         print('fc:',x.shape)
        return x


    def _decode_full_4layers(self, x, reuse=tf.AUTO_REUSE, use_res_block=False, is_training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            x = tf.reshape(x, [N, -1])
            print('decode:\n',x.shape)
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-2]*self.F[-1]))            # N x MF
                print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[-2]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(1, 4):
                with tf.variable_scope('upconv{}'.format(i+1)):
                    # with tf.name_scope('unpooling'):
                    #     x = self.unpool(x, self.U[-i-1])
                    #     print('unpool:',x.shape)
                    if not use_res_block:
                        with tf.name_scope('filter'):
                            # x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                            x = self.filter(x, self.L[0], self.F[-i-1], self.K[0])
                            print('filter:',x.shape)
                            #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
                        with tf.name_scope('bias_relu'):
                            x = self.brelu(x)
                            print('brelu:',x.shape)
                    else:
                        x = self.gcn_res_block_4layers(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)
                        # x = self.gcn_res_block_simple(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)

            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                # x = tf.tanh(x)
                print('refilter:',x.shape)
            # exit()
        return x



    def _decode_full_6890(self, x, reuse=tf.AUTO_REUSE, use_res_block=False, is_training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            #M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            print('decode:\n',x.shape)
            with tf.variable_scope('fc2'):
                # x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
                x = tf.layers.dense(x, int(self.F[-1]), activation=tf.nn.leaky_relu)
                print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[0]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(4):
                with tf.variable_scope('upconv{}'.format(i+1)):
                    # with tf.name_scope('unpooling'):
                    #     x = self.unpool(x, self.U[-i-1])
                    #     print('unpool:',x.shape)
                    if not use_res_block:
                        with tf.name_scope('filter'):
                            # x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                            x = self.filter(x, self.L[0], self.F[-i-1], self.K[0])
                            print('filter:',x.shape)
                            #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
                        with tf.name_scope('bias_relu'):
                            x = self.brelu(x)
                            print('brelu:',x.shape)
                    else:
                        x = self.gcn_res_block(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)
                        # x = self.gcn_res_block_simple(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)

            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                # x = tf.tanh(x)
                print('refilter:',x.shape)
            # exit()
        return x

    def _decode_without_latent(self, x, reuse=False):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            #M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            # print('decode:\n',x.shape)
            # with tf.variable_scope('fc2'):
            #     x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
            #     print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(len(self.F)):
                with tf.variable_scope('upconv{}'.format(i+1)):
                    with tf.name_scope('unpooling'):
                        x = self.unpool(x, self.U[-i-1])
                        print('unpool:',x.shape)
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                        print('filter:',x.shape)
                        #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                        print('brelu:',x.shape)

            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                print('refilter:',x.shape)
            # exit()
        return x

    def _decode_part1(self, x, reuse=False):
        with tf.variable_scope('decoder_part1', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            #M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            print('decode:\n',x.shape)
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
                print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(len(self.F)):
                with tf.variable_scope('upconv{}'.format(i+1)):
                    with tf.name_scope('unpooling'):
                        x = self.unpool(x, self.U[-i-1])
                        print('unpool:',x.shape)
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                        print('filter:',x.shape)
                        #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                        print('brelu:',x.shape)

            # with tf.name_scope('outputs'):
            #     x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
            #     print('refilter:',x.shape)
            # exit()
        return x


    def _decode_part2(self, x, reuse=False):
        with tf.variable_scope('decoder_part2'):
            # N = x.get_shape()[0]
            # #M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            # print('decode:\n',x.shape)
            # with tf.variable_scope('fc2'):
            #     x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
            #     print('fc:',x.shape)

            # x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F
            # print('reshape:',x.shape)

            # for i in range(len(self.F)):
            #     with tf.variable_scope('upconv{}'.format(i+1)):
            #         with tf.name_scope('unpooling'):
            #             x = self.unpool(x, self.U[-i-1])
            #             print('unpool:',x.shape)
            with tf.variable_scope('filter'):
                x = self.filter(x, self.L[0], self.F[0], self.K[0])
                print('filter:',x.shape)
                #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
            with tf.variable_scope('bias_relu'):
                x = self.brelu(x)
                print('brelu:',x.shape)

            with tf.variable_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                print('refilter:',x.shape)
            # exit()
        return x

    def _decode(self, x, use_resblock=True, reuse=False, is_training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            #M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            print('decode:\n',x.shape)
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
                print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(len(self.F)):
                if not use_resblock:
                    with tf.variable_scope('upconv{}'.format(i+1)):
                        with tf.name_scope('unpooling'):
                            x = self.unpool(x, self.U[-i-1])
                            print('unpool:',x.shape)
                        with tf.name_scope('filter'):
                            x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                            print('filter:',x.shape)
                            #print(self.L[-(i+1)], self.F[-(i+1)], self.K[-(i+1)])
                        with tf.name_scope('bias_relu'):
                            x = self.brelu(x)
                            print('brelu:',x.shape)
                else:
                    x = self.gcn_res_block_4layers(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)
            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                print('refilter:',x.shape)
            # exit()
        return x

    def group_normalizaton(self, x, is_training, name, norm_type='group', G=8, eps=1e-5, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            if norm_type == None:
                output = x 
            elif norm_type == 'batch':
                output = tf.contrib.layers.atch_norm(
                    x, center=True, scale=True, decay=0.999,
                    is_training=is_training, updates_collections=None
                )
            elif norm_type == 'group':
                # tranpose: [bs, v, c] to [bs, c, v] following the GraphCMR paper
                x = tf.transpose(x, [0, 2, 1])
                N, C, V = x.get_shape().as_list() # v is num of verts
                G = min(G, C)
                x = tf.reshape(x, [-1, G, C // G, V])
                mean, var = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
                x = (x -mean) / tf.sqrt(var + eps)
                # per channel gamma and beta
                gamma = tf.get_variable('gamma', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                beta  =tf.get_variable('beta', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                gamma = tf.reshape(gamma, [1, C, 1])
                beta = tf.reshape(beta, [1, C, 1])

                output = tf.reshape(x, [-1, C, V]) * gamma + beta

                output = tf.transpose(output, [0, 2, 1])

            else:
                raise NotImplementedError

        return output

    def gcn_res_block(self, x_in, i, name, reuse=False, is_training=False):
        with tf.variable_scope(name, reuse=reuse):
            x = self.group_normalizaton(x_in, is_training=is_training, name='group_norm', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_linear_1'):
                x = self.filter(x, self.L[0], self.F[-i-1] // 2, 1)
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_1', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.L[0], self.F[-i-1] // 2, self.K[0])
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_2', reuse=reuse)

            x = tf.nn.relu(x)
            
            with tf.variable_scope('graph_linear_2'):
                x = self.filter(x, self.L[0], self.F[-i-1], 1)
            
            channel_in = x_in.get_shape()[-1]
            channel_out = x.get_shape()[-1]
            if channel_in != channel_out:
                with tf.variable_scope('graph_linear_input'):
                    x_in = self.filter(x_in, self.L[0], channel_out, 1)

            # skip connection
            x = x + x_in

        return x
    
    def gcn_res_block_4layers(self, x_in, i, name, reuse=False, is_training=False):
        with tf.variable_scope(name, reuse=reuse):

            with tf.name_scope('unpooling'):
                x_in = self.unpool(x_in, self.U[-i-1])

            x = self.group_normalizaton(x_in, is_training=is_training, name='group_norm', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_linear_1'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1] // 2, 1)
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_1', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1] // 2, self.K[0])
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_2', reuse=reuse)

            x = tf.nn.relu(x)
            
            with tf.variable_scope('graph_linear_2'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1], 1)
            
            channel_in = x_in.get_shape()[-1]
            channel_out = x.get_shape()[-1]
            if channel_in != channel_out:
                with tf.variable_scope('graph_linear_input'):
                    x_in = self.filter(x_in, self.L[-i-2], channel_out, 1)

            # skip connection
            x = x + x_in

        return x

    def gcn_res_block_simple(self, x_in, i, name, reuse=False, is_training=False):
        with tf.variable_scope(name, reuse=reuse):
            x = self.group_normalizaton(x_in, is_training=is_training, name='group_norm', reuse=reuse)

            # x = tf.nn.relu(x)

            # with tf.variable_scope('graph_linear_1'):
            #     x = self.filter(x, self.L[0], self.F[-i-1] // 2, 1)
            # x = self.group_normalizaton(x, is_training=is_training, name='group_norm_1', reuse=reuse)

            # x = tf.nn.relu(x)

            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.L[0], self.F[-i-1] // 2, self.K[0])
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_2', reuse=reuse)

            # x = tf.nn.relu(x)
            
            # with tf.variable_scope('graph_linear_2'):
            #     x = self.filter(x, self.L[0], self.F[-i-1], 1)
            
            channel_in = x_in.get_shape()[-1]
            channel_out = x.get_shape()[-1]
            if channel_in != channel_out:
                with tf.variable_scope('graph_linear_input'):
                    x_in = self.filter(x_in, self.L[0], channel_out, 1)

            # skip connection
            x = x + x_in

        return x







    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph, config=self.config)
            
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def decode(self,data):
        size = data.shape[0]
        x_rec = [0]*size
        sess = self._get_session(sess=None)
        with sess:
            # load params of part model
            self.allvariable = tf.trainable_variables()
            self.decode_variable = [var for var in self.allvariable if 'decoder' in var.name.split('/')[0]]            
            self.op_saver = tf.train.Saver(self.decode_variable)
            self.op_saver.restore(sess, self.dir_name)


            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                end = min([end, size])
                
                batch_data = np.zeros((self.batch_size, data.shape[1]))
                tmp_data = data[begin:end,:]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
                feed_dict = {self.ph_z: batch_data, self.ph_dropout: 1}
                
                batch_pred = sess.run(self.op_decoder, feed_dict)
                
                x_rec[begin:end] = batch_pred[:end-begin]
            
            x_rec = np.array(x_rec)
            return x_rec



def store_sp_matrix(list_of_matrix,  name):
    """
    Param:
        list_of_matrix: A list of sparse matrix.
        name: The name of matrix needed to store.
    
    """
    dir_name = os.getcwd()
    dir_name = os.path.join(dir_name, 'matrix', name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for i in range(len(list_of_matrix)):
        assert(scipy.sparse.issparse(list_of_matrix[i]))

        abs_dir_name = dir_name + '/' + str(i) + '.npz'
        scipy.sparse.save_npz(abs_dir_name, list_of_matrix[i])




def read_sp_matrix(name):
    """
    Param:
        name: The name of matrix needed to read.

    Return:
        A list of sparse matrix.

    """
    dir_name = os.path.dirname(os.path.realpath(__file__)) 
    dir_name = os.path.join(dir_name, name)
    sp_matrix = []
    list_dir = os.listdir(dir_name)
    for i in range(len(list_dir)):
        dir_name_npz = dir_name + '/' + str(i) + '.npz'
        sparse_matrix_variable = scipy.sparse.load_npz(dir_name_npz)
        sp_matrix.append(sparse_matrix_variable)

    return sp_matrix