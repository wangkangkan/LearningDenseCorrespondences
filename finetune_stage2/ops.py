"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def keypoint3D_l1_loss_select(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))

        kp_pred = tf.reshape(kp_pred, (-1, 3))
        kp_pred_select = tf.gather(kp_pred[kp_gt[:, 3], :])

        res = tf.losses.absolute_difference(kp_gt[:, :3], kp_pred_select)

        return res

def keypoint3D_loss_select(kp_gt, kp_pred, batchsidx, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))
        kp_pred = tf.reshape(kp_pred, (-1, 3))
        skp_gt = tf.gather(kp_gt, batchsidx)
        skp_pred = tf.gather(kp_pred, batchsidx)
        svis = tf.cast(tf.reshape(skp_gt[:, 3],[-1]), tf.float32)
        td = skp_gt[:, 0:3]-skp_pred
        distsquare = tf.reduce_sum(tf.multiply(td,td), 1)
        res = tf.reduce_sum(tf.multiply(svis, distsquare))
        res = res / tf.reduce_sum(svis)
        return res

def keypoint3D_loss(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 3))
        # vis = tf.cast(tf.reshape(kp_gt[:, 3],[-1]), tf.float32)
        td = kp_gt[:, 0:3]-kp_pred
        distsquare = tf.reduce_sum(tf.multiply(td,td), 1)
        res = tf.reduce_sum(distsquare)
        #res = res/tf.reduce_sum(vis)
        return res

def keypoint3D_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))
        kp_pred = tf.reshape(kp_pred, (-1, 3))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 3], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :3], kp_pred, weights=vis)
        return res

def keypoint_l1_loss_projection(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    campro = tf.constant([ [fx_d,0,cx_d], [0,fy_d,cy_d], [0,0,1]])
    campro = tf.transpose(campro,[1,0])
    with tf.name_scope(name, "keypoint_l1_loss_projection", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 3))#x,y,z

        projkp = tf.matmul(kp_pred, campro)
        proi = tf.reshape(tf.div(projkp[:, 0], projkp[:, 2]),[-1,1])
        proj = tf.reshape(tf.div(projkp[:,1], projkp[:,2]),[-1,1])
        projkp2d = tf.concat([proi, proj],1)

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        res = tf.losses.absolute_difference(kp_gt[:, :2], projkp2d, weights=vis)
        return res

def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res

def compute_3d_loss_our(params_pred, params_gt):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt]):
        # res = tf.losses.mean_squared_error(
        #     params_gt, params_pred) * 0.5
        res = tf.reduce_sum(tf.square(params_gt-params_pred))
        return res

def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = tf.losses.mean_squared_error(
            params_gt, params_pred, weights=weights) * 0.5
        return res


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)



def nearestpoint_distance_and_normal_model(points1, point2, k, pointsnormal1, pointnormal2):
    
    Y1 = points1#(batch_size, num_points1, 3)
    Y2 = point2#(batch_size, num_points2, 3)
    Y1T = tf.transpose(Y1, perm=[0, 2, 1])
    Y2T = tf.transpose(Y2, perm=[0, 2, 1])
    Y3 = tf.matmul(tf.multiply(Y1, Y1), tf.ones(tf.shape(Y2T))) + tf.matmul(tf.ones(tf.shape(Y1)),tf.multiply(Y2T, Y2T)) - tf.multiply(2.0, tf.matmul(Y1, Y2T))
    distance = tf.sqrt(Y3, name='match_relation_matrix')#(batch_size, num_points1, num_points2)
    dot_product = tf.matmul(pointsnormal1,pointnormal2,transpose_b=True)
    # index_mask = tf.where(dot_product>0, )
    Nmatmul = tf.maximum(dot_product, 0)
    neg_adj = tf.maximum(1-distance/0.1,0)
    temp = tf.multiply(Nmatmul,neg_adj)

    topk, nn_idx = tf.nn.top_k(temp, k=k)
    temp_for_mask = tf.reduce_sum(dot_product, axis=-1)
    ones = tf.ones_like(temp_for_mask)
    zeros = tf.zeros_like(temp_for_mask)
    mask = tf.where(temp_for_mask>0,ones,zeros)
    # distance_with_normal = tf.reduce_mean(tf.reduce_sum(topk, axis=[1, 2]))
    return nn_idx, mask

# from evaluatetest import POINT_NUMBER
def normal_loss_model(points1, point2, pointsnormal1, pointsnormal2, k=1):
    nn_idx, mask = nearestpoint_distance_and_normal_model(points1, point2, k, pointsnormal1, pointsnormal2)
    # print(nn_idx)
    # print(point2)
    # exit()
    batch_size = point2.get_shape().as_list()[0]
    num_point = point2.get_shape().as_list()[1]
    POINT_NUMBER = points1.get_shape().as_list()[1]

    idx_ = tf.range(batch_size) * num_point
    idx_ = tf.reshape(idx_, [batch_size, 1])
    nn_idx = tf.reshape(nn_idx,[batch_size,-1])
    IDX= nn_idx+idx_
    IDX = tf.reshape(IDX,[-1,1])
    predictvertsnew = tf.reshape(point2,[batch_size*num_point,-1])
    predictvertsnew = tf.gather(predictvertsnew,IDX)
    predictvertsnew = tf.reshape(predictvertsnew,[batch_size,POINT_NUMBER,1,-1])
    gather_pre_vert = tf.squeeze(predictvertsnew,axis=2)
    # gather_pre_vert = tf.gather_nd(point2, nn_idx)

    # loss = tf.reduce_mean(tf.reduce_sum(mask * tf.sqrt(tf.reduce_sum((points1 - gather_pre_vert)**2, axis=[2])), axis=1))
    loss = tf.reduce_mean(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(points1 - gather_pre_vert), axis=[2]), axis=[1]))
    
    # distance2 = nearestpoint_distance_and_normal(point2, points1, k, pointsnormal2, pointsnormal1)

    return loss



def cal_normals(vert, face):
    batch_size, point_num, channel = vert.get_shape().as_list()
    new_vert_0 = tf.gather(vert, indices=face[:, 0], axis=1)
    new_vert_1 = tf.gather(vert, indices=face[:, 1], axis=1)
    new_vert_2 = tf.gather(vert, indices=face[:, 2], axis=1)

    normal = tf.cross(new_vert_1 - new_vert_0, new_vert_2 - new_vert_0, name="normals")

    normal = tf.nn.l2_normalize(tf.reduce_mean(tf.reshape(normal, [batch_size, point_num, -1, channel]), axis=2), axis=-1)
    return normal





def Laplacian_loss(kp_loader, pred, con):
    grouped_kp_loader = tf.transpose(kp_loader, [1, 0, 2])
    grouped_kp_loader = tf.gather(grouped_kp_loader, con)
    grouped_kp_loader = tf.transpose(grouped_kp_loader, [2, 0, 1, 3])
    # print(grouped_kp_loader)
    grouped_pred = tf.transpose(pred, [1, 0, 2])
    grouped_pred = tf.gather(grouped_pred, con)
    grouped_pred = tf.transpose(grouped_pred, [2, 0, 1, 3])
    # print(grouped_pred)
    # kp_loader = tf.reshape(kp_loader, (-1, 3))
    # pred = tf.reshape(pred, (-1, 3))
    td1 = kp_loader-pred
    res1 = tf.multiply(td1,td1)
    # print(res1)
    td2 = grouped_kp_loader-grouped_pred
    res2 = tf.reduce_mean(tf.multiply(td2,td2),axis=2)
    # print(res2)
    td = res1-res2
    res = tf.reduce_sum(tf.multiply(td,td))
    print(res)
    # exit()
    return res


def edge_loss(kp_loader, pred, edge):
    # batch_size = kp_loader.get_shape()[0].value
    # num_point = kp_loader.get_shape()[1].value
    idx1 = edge[:,0]
    # idx1 = edge[0,:]
    idx2 = edge[:,1]
    # idx2 = edge[1,:]
    kp_loader1 = tf.transpose(kp_loader,[1,0,2])
    kp_loader1 = tf.gather(kp_loader1,idx1)
    kp_loader1 = tf.transpose(kp_loader1,[1,0,2])
    kp_loader2 = tf.transpose(kp_loader,[1,0,2])
    kp_loader2 = tf.gather(kp_loader2,idx2)
    kp_loader2 = tf.transpose(kp_loader2,[1,0,2])
    td1 = kp_loader1-kp_loader2
    res1 = tf.multiply(td1,td1)
    pred1 = tf.transpose(pred,[1,0,2])
    pred1 = tf.gather(pred1,idx1)
    pred1 = tf.transpose(pred1,[1,0,2])
    pred2 = tf.transpose(pred,[1,0,2])
    pred2 = tf.gather(pred2,idx2)
    pred2 = tf.transpose(pred2,[1,0,2])
    td2 = pred1-pred2
    res2 = tf.multiply(td2,td2)
    td = res1-res2
    res = tf.reduce_sum(tf.multiply(td,td))
    print(res)
    # exit()
    return res

def edge_loss_2(points, faces, template):
    # sommet_A = points[:, faces[:, 0], :]
    # sommet_B = points[:,faces[:, 1], :]
    # sommet_C = points[:,faces[:, 2], :]
    
    sommet_A = tf.gather(points, indices=faces[:, 0], axis=1)
    sommet_B = tf.gather(points, indices=faces[:, 1], axis=1)
    sommet_C = tf.gather(points, indices=faces[:, 2], axis=1)

    score = tf.abs(tf.sqrt(tf.reduce_sum((sommet_A - sommet_B) ** 2, axis=-1)) / template[0] -1)
    score += tf.abs(tf.sqrt(tf.reduce_sum((sommet_C - sommet_B) ** 2, axis=-1)) / template[1] -1)
    score += tf.abs(tf.sqrt(tf.reduce_sum((sommet_A - sommet_C) ** 2, axis=-1)) / template[2] -1)

    return tf.reduce_mean(score)