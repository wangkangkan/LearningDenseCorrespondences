from . import rescale

import tensorflow as tf
import scipy.sparse
import numpy as np
import os

class DecodeModel(object):
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

    def _decode(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('decoder', reuse=reuse):
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

            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                print('refilter:',x.shape)
            # exit()
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