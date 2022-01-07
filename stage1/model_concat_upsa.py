import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils.pointnet_util import *
from lib.decodeModel import DecodeModel, read_sp_matrix


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(40000)
BN_DECAY_CLIP = 0.99


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 104))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, masks_pl


def get_model(point_cloud, idx_1723_tf=None, is_training=True, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud['pointcloud'].get_shape()[0].value
    num_point = 6890

    l0_xyz_f1 = point_cloud['pointcloud'][:, :num_point, 0:3]

    l0_xyz_f2 = point_cloud['pointcloud'][:, num_point:, 0:3]

    l0_xyz_f1 = tf.gather(l0_xyz_f1, indices=idx_1723_tf, axis=1)

    decode_model = get_class()

    down_sample = []
    for i in range(4):
        down_file_name = './lib/Down_sample/'+str(i)+'.txt'
        # print(down_file_name)
        down = np.loadtxt(down_file_name).astype(np.int)
        down_sample.append(down[:, 1])

    all_data = {}
    all_data['frm1'] = l0_xyz_f1 # (4, 1723, 3)
    all_data['frm2'] = l0_xyz_f2 # (4, 2000, 3)

    RADIUS1 = 0.05
    RADIUS2 = 0.1
    RADIUS3 = 0.2
    RADIUS4 = 0.4

    l1_xyz_f1 = down_sample_fun_simple(xyz=l0_xyz_f1, index=down_sample[0])
    l2_xyz_f1 = down_sample_fun_simple(xyz=l1_xyz_f1, index=down_sample[1])
    l3_xyz_f1 = down_sample_fun_simple(xyz=l2_xyz_f1, index=down_sample[2])
    l4_xyz_f1 = down_sample_fun_simple(xyz=l3_xyz_f1, index=down_sample[3])

    l4_points_f1 = decode_model._encode(l0_xyz_f1)
    
    with tf.variable_scope('sa1',  reuse=tf.AUTO_REUSE) as scope:

        
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, points=None, npoint=1500, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        # Frame 2, Layer 2
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, points=l1_points_f2, npoint=500, radius=RADIUS2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        
        l3_xyz_f2, l3_points_f2, l3_indices_f2 = pointnet_sa_module(l2_xyz_f2, points=l2_points_f2, npoint=100, radius=RADIUS3, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l3_xyz_f2, l3_points_f2, l3_indices_f2 = pointnet_sa_module(l3_xyz_f2, points=l3_points_f2, npoint=50, radius=RADIUS4, nsample=64, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

        l4_xyz_f2, l4_points_f2, l4_indices_f2 = pointnet_sa_module(l3_xyz_f2, points=l3_points_f2, npoint=100, radius=RADIUS4, nsample=16, mlp=[512,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer_global')

    latent1 = decode_model._encode_fc(l4_points_f1, nz=1024)
    latent2 = tf.reshape(l4_points_f2, shape=[batch_size, -1])
    latent = tf.concat([latent1, latent2], axis=1)
    
    net = decode_model._decode(latent)

    return net, all_data

def get_class():

    '''read matrix '''
    A = read_sp_matrix('A')
    A = list(map(lambda x: x.astype('float32'), A))  # float64 -> float32
    U = read_sp_matrix('U')
    U = list(map(lambda x: x.astype('float32'), U))  # float64 -> float32
    D = read_sp_matrix('D')
    D = list(map(lambda x: x.astype('float32'), D))  # float64 -> float32
    L = read_sp_matrix('L')

    p = list(map(lambda x: x.shape[0], A))  # [6890, 1723, 431, 108, 27]

    # print('p', np.shape(p[0]))

    params = dict()
    # Architecture.
    nz = 256  # 512
    params['F_0'] = 3  # Number of graph input features.
    params['F'] = [32, 64, 128, 256]  # Number of graph convolutional filters.
    params['K'] = [2, 2, 2, 2]  # Polynomial orders.

    # pretrained_decodermodel_path = './dataset/decodermodel/model-112670'
    decodermodel = DecodeModel(L, D, U, params['F'], params['K'], p, nz, F_0=params['F_0'])

    return decodermodel


def get_loss(pred, label, idx_1723_tf=None):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """
    if idx_1723_tf is not None:
        label = tf.gather(label, indices=idx_1723_tf, axis=1)


    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    channel = label.get_shape()[2].value

    if channel == 4:
        mask = label[:, :, -1]
        label = label[:, :, :3]
        # print('use')
    else:
        assert (channel == 3)
        mask = tf.constant(1.0, dtype=tf.float32, shape=[batch_size, num_point], name='mask')

    l2_loss = tf.reduce_mean(tf.reduce_sum(mask * tf.sqrt(tf.reduce_sum((pred-label)**2, axis=2)), axis=1) / tf.reduce_sum(mask, axis=1))

    return l2_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
