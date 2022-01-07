import tensorflow as tf
import tf_util
from transform_nets import input_transform_net # , feature_transform_net

def pointnet_feature(point_cloud, is_training, bn_decay=None):
    # print('is_training------------------------------------->', is_training)
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1
    
    with tf.variable_scope("pointnet") as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)

        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # net = tf_util.conv2d(net, 2048, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv6', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
        
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])

        #net = tf.sigmoid(net)

        tmp = net
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        # net = tf_util.fully_connected(net, 2048, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables #, tmp
