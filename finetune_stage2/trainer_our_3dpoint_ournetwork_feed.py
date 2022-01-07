from data_loader_our_3dpoint import num_examples

from ops import keypoint3D_loss, compute_3d_loss_our, normal_loss_model, cal_normals, edge_loss, Laplacian_loss
from models import Discriminator_separable_rotations, get_encoder_fn_separate

from tf_smpl.batch_lbs import batch_rodrigues
from tf_smpl.batch_smpl_our import SMPL

from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np
from lib.decodeModel import DecodeModel, read_sp_matrix

from os.path import join, dirname, split
from write2obj import write_to_obj

import glob
import random

import pointnet2

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
    # params['F'] = [32, 64, 128, 256]  # Number of graph convolutional filters.
    params['F'] = [16, 32, 64, 64]  # Number of graph convolutional filters.
    params['K'] = [4,4,4,4]  # Polynomial orders.

    # pretrained_decodermodel_path = './dataset/decodermodel/model-112670'
    decodermodel = DecodeModel(L, D, U, params['F'], params['K'], p, nz, F_0=params['F_0'])

    return decodermodel


class HMRTrainer(object):
    def __init__(self, config, data_loader, mocap_loader):
        """
        Args:
          config
          if no 3D label is available,
              data_loader is a dict
          else
              data_loader is a dict
        mocap_loader is a tuple (pose, shape)
        """
        # Config + path
        self.config = config
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        self.encoder_only = config.encoder_only
        self.use_3d_label = config.use_3d_label

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch

        self.num_cam = 3#translation

        #including R
        self.num_theta = 72 * 2 # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Data
        num_images = 1248
        num_mocap = num_examples(config.mocap_datasets)

        self.num_itr_per_epoch = int(num_images / self.batch_size)
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        self.pointset_loader = tf.placeholder(tf.float32, shape=[self.batch_size, 3000, 3], name="input_pc")
        self.pointset_loader_normal = tf.placeholder(tf.float32, shape=[self.batch_size, 3000, 3], name="input_pc_normal")
        self.kp_loader = tf.placeholder(tf.float32, shape=[self.batch_size, 6890, 3], name="label") #+ self.pointset_loader#3D point locations
        self.pred_smpl = tf.placeholder(tf.float32, shape=[self.batch_size, 6890, 3], name="stage3")
        self.paras = tf.placeholder(tf.float32, shape=[self.batch_size, 157], name='smpl_paras')
        self.gt_joints = tf.placeholder(tf.float32, shape=[self.batch_size, 14, 3], name='joint')
        self.joint_mask = tf.placeholder(tf.float32, shape=[self.batch_size, 14], name='joint_mask')

        conname = './con_4.txt'
        self.con = np.loadtxt(conname, dtype=np.int)-1
        edgename = './edge2.txt'
        self.edge = np.loadtxt(edgename, dtype=np.int)-1
        face = np.loadtxt('./normal_face_6890_x6.txt', dtype=np.int) - 1
        self.faces = tf.constant(face, dtype=tf.int32)

        self.count_for_save = 0

        MPIJidx = [15,17,19,21,16,18,20,2,5,8,1,4,7,6]
        self.MPIJidx = tf.constant(np.array(MPIJidx), dtype=tf.int32, name="joint_index")

        sidx = np.arange(0, 6890, 20)
        allsidx = sidx
        for i in range(config.batch_size-1):
            allsidx = np.concatenate([allsidx, sidx + 6890 * (i + 1)], 0)
        self.batchsidx = tf.constant(sidx)

        self.pose_loader = mocap_loader[0]
        self.shape_loader = mocap_loader[1]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.log_img_step = config.log_img_step

        # Model spec
        self.model_type = config.model_type
        self.keypoint_loss = keypoint3D_loss

        # Optimizer, learning rate
        self.e_lr = config.e_lr
        self.d_lr = config.d_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd
        self.e_loss_weight = config.e_loss_weight
        self.d_loss_weight = config.d_loss_weight
        self.e_3d_weight = config.e_3d_weight

        self.optimizer = tf.train.AdamOptimizer

        self.decodemolde = get_class()


        down_file_name = '0.txt' 
        down = np.loadtxt(down_file_name).astype(np.int)
        idx_1723 = down[:, 1]
        self.idx_1723_tf = tf.constant(idx_1723, dtype=tf.int32)


        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)
        self.E_var = []
        self.build_model()

        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.pretrained_model_path)
            if 'resnet_v2_50' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v2_50' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            elif 'pose-tensorflow' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v1_101' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            else:
                
                # var_list = [var for var in tf.global_variables() if "detail_decode" not in var.name]
                # self.pre_train_saver = tf.train.Saver(var_list)
                self.pre_train_saver = tf.train.Saver()
                # exit()
            def load_pretrain(sess):
                self.pre_train_saver.restore(sess, self.pretrained_model_path)

            init_fn = load_pretrain

        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=5)
        # self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            global_step=self.global_step,
            saver=self.saver,
            # summary_writer=self.summary_writer,
            init_fn=init_fn)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        if ('resnet' in self.model_type) and (self.pretrained_model_path is
                                              not None):
            # Check is model_dir is empty
            import os
            if os.listdir(self.model_dir) == []:
                return True

        return True

    def l1_loss(self, pred, label):
        return tf.reduce_mean(tf.reduce_sum(self.joint_mask * tf.sqrt(tf.reduce_sum((pred-label[:, :, :3])**2, axis=-1)), axis=1), axis=0)

    def build_model(self):

        def rot6d_to_rotmat(x):
            """Convert 6D rotation representation to 3x3 rotation matrix.
            Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
            Input:
                (B,6) Batch of 6-D rotation representations
            Output:
                (B,3,3) Batch of corresponding rotation matrices
            """
            x = tf.reshape(x, [-1,3,2])
            a1 = x[:, :, 0]
            a2 = x[:, :, 1]
            b1 = tf.nn.l2_normalize(a1,dim=1)
            b2 = tf.nn.l2_normalize(a2 - tf.expand_dims(tf.einsum('bi,bi->b', b1, a2),-1) * b1, dim=1)
            b3 = tf.cross(b1, b2)
            return tf.concat([b1, b2, b3], 1)

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)

        self.img_feat, self.E_var= pointnet2.extract_globalfeature(self.pointset_loader, None, tf.cast(True, tf.bool))
        self.smpl_feat, self.E_var_smpl= pointnet2.pointnet_feature(self.pred_smpl, tf.cast(True, tf.bool))
        # self.E_var.extend(self.E_var_smpl)

        loss_kps = []
        loss_personal = []
        loss_3d_params = []
        # For discriminator
        fake_rotations, fake_shapes = [], []
        # Start loop
        # 85D
        # pred_pose, pred_shape, pred_cam = self.load_mean_param1()
        pred_pose = self.paras[:, :144]
        pred_shape = self.paras[:, 144:154]
        pred_cam = self.paras[:, 154:]
        # self.E_var.extend([pred_pose, pred_shape, pred_cam])

        # For visualizations
        self.all_verts = []
        self.all_pred_kps = []
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, self.smpl_feat, pred_pose, pred_shape, pred_cam], 1)
            # state = tf.concat([self.img_feat, tf.reshape(self.pred_smpl, [self.batch_size, -1]), pred_pose, pred_shape, pred_cam], 1)

            if i == 0:
                delta_pose, delta_shape, delta_cam, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_pose, delta_shape, delta_cam, _ = threed_enc_fn(
                    state, num_output=self.total_params, reuse=True)

            # Compute new theta
            # theta_here = theta_prev + delta_theta
            pred_pose = pred_pose + delta_pose
            pred_shape = pred_shape + delta_shape
            pred_cam = pred_cam + delta_cam
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            pred_rotmat = tf.reshape(rot6d_to_rotmat(pred_pose), [self.batch_size, 24, 3, 3])
            
            verts, pred_Rs, self.pred_joints = self.smpl(pred_shape, pred_rotmat)

            pred_kp = verts + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating
            self.pred_joints = self.pred_joints + tf.reshape(tf.tile(pred_cam,[1,24]),[-1,24,3])#tranlating

            pred_kp_1723 = tf.gather(pred_kp, self.idx_1723_tf, axis=1)
            
            # --- Compute losses:
            loss_kps.append(self.e_loss_weight * self.keypoint_loss(
                self.kp_loader, pred_kp))

            self.all_theta_prev.append(tf.concat([pred_pose, pred_shape, pred_cam], axis=1))
            self.all_pred_kps.append(pred_kp)

            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])

            # Save pred_rotations for Discriminator
            fake_rotations.append(pred_Rs[:, 1:, :])
            fake_shapes.append(pred_shape)

        self.pred_joints = tf.gather(self.pred_joints ,self.MPIJidx, axis=1)
        pred_kp_normal = cal_normals(pred_kp, self.faces)
        self.pred_smpl_1723 = tf.gather(self.pred_smpl, self.idx_1723_tf, axis=1)
        self.normal_loss = normal_loss_model(self.pointset_loader, pred_kp, self.pointset_loader_normal, pred_kp_normal)
        self.edge_loss = edge_loss(self.pred_smpl_1723, pred_kp_1723, self.edge)
        self.laplacian_loss = Laplacian_loss(self.pred_smpl_1723, pred_kp_1723, self.con)

        self.joints_loss = self.l1_loss(self.pred_joints, self.gt_joints)

        if not self.encoder_only:
            self.setup_discriminator(fake_rotations, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_e_loss"):
            # Just the last loss.
            self.e_loss_kp = loss_kps[-1]

            if self.encoder_only:
                self.e_loss = self.e_loss_kp #+ self.e_loss_weight * self.e_loss_kp_personal
            else:
                self.e_loss = self.d_loss_weight * self.e_loss_disc + self.e_loss_weight  * self.normal_loss + self.laplacian_loss + self.edge_loss + self.e_loss_weight * self.joints_loss * 0.01

            self.e_loss_3d = tf.constant(0)
        if not self.encoder_only:
            with tf.name_scope("gather_d_loss"):
                self.d_loss = self.d_loss_weight * (
                    self.d_loss_real + self.d_loss_fake)

        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)

        # Setup optimizer
        print('Setting up optimizer..')
        d_optimizer = self.optimizer(self.d_lr)
        e_optimizer = self.optimizer(self.e_lr)

        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.global_step, var_list=self.E_var)
        if not self.encoder_only:
            self.d_opt = d_optimizer.minimize(self.d_loss, var_list=self.D_var)

        print('Done initializing trainer!')


    def setup_discriminator(self, fake_rotations, fake_shapes):
        # Compute the rotation matrices of "rea" pose.
        # These guys are in 24 x 3.
        real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
        real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
        # Ignoring global rotation. N x 23*9
        # The # of real rotation is B*num_stage so it's balanced.
        real_rotations = real_rotations[:, 1:, :]
        all_fake_rotations = tf.reshape(
            tf.concat(fake_rotations, 0),
            [self.batch_size * self.num_stage, -1, 9])
        comb_rotations = tf.concat(
            [real_rotations, all_fake_rotations], 0, name="combined_pose")

        comb_rotations = tf.expand_dims(comb_rotations, 2)
        all_fake_shapes = tf.concat(fake_shapes, 0)
        comb_shapes = tf.concat(
            [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.d_wd,
            'shapes': comb_shapes,
            'poses': comb_rotations
        }

        self.d_out, self.D_var = Discriminator_separable_rotations(
            **disc_input)

        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)
        # Compute losses:
        with tf.name_scope("comp_d_loss"):
            self.d_loss_real = tf.reduce_mean(
                tf.reduce_sum((self.d_out_real - 1)**2, axis=1))
            self.d_loss_fake = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake)**2, axis=1))
            # Encoder loss
            self.e_loss_disc = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake - 1)**2, axis=1))

    def get_3d_loss_our(self, Rs, shape, cam_Rs, translation):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        tcam_Rs = tf.reshape(cam_Rs, [self.batch_size, -1])
        tRs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([tRs, shape, tcam_Rs, translation], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshapetranslation_loader[:, :238]
        loss_poseshapetranslation = self.e_3d_weight * compute_3d_loss_our(
            params_pred, gt_params)

        return loss_poseshapetranslation

    def load_joint(self, filepath):
        realjoint  = np.loadtxt(filepath)
        realjoint = realjoint[1:15,:]
        idx = np.where(realjoint[:,2]!=0, np.ones(14), np.zeros(14))
        # idx = idx[0]
        # idx = idx.tolist()
        # realjoint = realjoint[idx,:]
        return realjoint, idx


    def save_result(self, pred_kp, name):
        for i in range(self.batch_size):
            write_to_obj(self.dir + "stage4/{}.obj".format(name[i]), pred_kp[i])
            self.count_for_save += 1

    def train_feed(self):

        self.dir = "../hks_part3_faust/cape/cape_test/"
        step = 0
        # count = 0
        epoche_loss = 0
        epochd_loss = 0
        epochk_loss = 0
        epochloss_3d = 0
        epoch_person_loss = 0.
        iter = 0

        with self.sv.managed_session(config=self.sess_config) as sess:
            # sess.run(tf.global_variables_initializer())
            all_list = sorted(glob.glob(self.dir + "stage3/*.txt")) 
            mmm = len(all_list)
            input_pc = np.zeros((self.batch_size, 3000, 3))
            input_pc_normal = np.zeros((self.batch_size, 3000, 3))
            label = np.zeros((self.batch_size, 6890, 3))
            stage3 = np.zeros((self.batch_size, 6890, 3))
            smpl_paras = np.zeros((self.batch_size, 157))
            gt_joints = np.zeros((self.batch_size, 14, 3))
            gt_joints_mask = np.zeros((self.batch_size, 14))
            while not self.sv.should_stop():
                random.shuffle(all_list)
                num_iter = len(all_list) // self.batch_size

                feed_dict = {
                    self.pointset_loader : input_pc,
                    self.pointset_loader_normal : input_pc_normal,
                    self.kp_loader : label,
                    self.pred_smpl : stage3,
                    self.paras : smpl_paras,
                    self.gt_joints : gt_joints,
                    self.joint_mask : gt_joints_mask,
                }

                fetch_dict = {
                    "step": self.global_step,
                    "e_loss": self.e_loss,
                    # The meat
                    "e_opt": self.e_opt,
                    "loss_kp": self.e_loss_kp,
                    # "loss_person": self.e_loss_kp_personal,
                    #"feat": self.img_feat
                }
                if not self.encoder_only:
                    fetch_dict.update({
                        # For D:
                        "d_opt": self.d_opt,
                        "d_loss": self.d_loss,
                        "loss_disc": self.e_loss_disc,
                    })
                fetch_dict.update({
                    "loss_3d_params": self.e_loss_3d
                })
                ###############################################
                fetch_dict.update({
                    "normal":self.normal_loss,
                    "edge": self.edge_loss,
                    "lap": self.laplacian_loss,
                    "joint": self.joints_loss,
                })

                ########################### for save results ################
                fetch_dict.update({
                    "pointset": self.pred_smpl,
                    'label': self.kp_loader,
                    'input_pc': self.pointset_loader,
                    'paras': self.all_theta_prev[-1],
                    'pred_kp': self.all_pred_kps[-1],
                    'joints': self.gt_joints,
                    'pred_joints': self.pred_joints,
                })

                for i in range(num_iter):
                    names = []
                    t0 = time()
                    for j in range(self.batch_size):
                        name = split(all_list[i*self.batch_size + j])[1].rstrip('.txt')
                        input_pc[j] = np.loadtxt(self.dir + 'input_pc/{}.txt'.format(name))
                        input_pc_normal[j] = np.loadtxt(self.dir + 'input_pc/{}_normal.txt'.format(name))
                        label[j] = np.loadtxt(self.dir + 'label/{}.txt'.format(name))
                        stage3[j] = np.loadtxt(self.dir + 'stage3/{}.txt'.format(name))
                        smpl_paras[j] = np.loadtxt(self.dir + 'paras/{}.txt'.format(name))
                        gt_joints[j], gt_joints_mask[j] = self.load_joint(self.dir + 'pred_joint_cor/{}.txt'.format(name))
                        # gt_joints[j], gt_joints_mask[j] = self.load_joint(self.dir + 'pred_joints/{}.txt'.format(name))
                        names.append(name)
                    result = sess.run(fetch_dict, feed_dict)
                    t1 = time()

                    e_loss = result['e_loss']
                    step = result['step']
                    result['loss_person'] = 0
                    epoch = float(step) / self.num_itr_per_epoch
                    if self.encoder_only:
                        d_loss = 0
                        loss_3d = 0
                        loss_person = result['loss_person']
                        loss_kp = e_loss - 60*loss_person
                        print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, kp_loss: %.4f, loss_person:%.3f" %
                            (step, epoch, t1 - t0, e_loss, loss_kp, loss_person))
                    else:
                        d_loss = result['d_loss']
                        loss_kp = result['loss_kp']
                        loss_person = result['loss_person']
                        loss_3d = result['loss_3d_params']
                        print(
                            "itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f, kp_loss: %.4f, normal:%.3f, joint:%.3f, lap: %.4f, edge: %.4f"
                            % (step, epoch, t1 - t0, e_loss, d_loss, loss_kp, result['normal'], result['joint'], result['lap'], result['edge']))

                    epoche_loss = epoche_loss + e_loss
                    epochd_loss = epochd_loss + d_loss
                    epochk_loss = epochk_loss + loss_kp
                    epoch_person_loss += loss_person
                    epochloss_3d = epochloss_3d+ loss_3d
                    iter = iter + 1
                    if step % self.num_itr_per_epoch == 0 and step >= 1:
                        # self.saver.save(sess,'./logs/models/model.ckpt', global_step = int(epoch))
                        f = open('./loss.txt', 'a+')
                        epoche_loss = epoche_loss / iter
                        epochd_loss = epochd_loss / iter
                        epochk_loss = epochk_loss / iter
                        epochloss_3d = epochloss_3d / iter
                        epoch_person_loss /= iter
                        f.write(str("%d %.5f %.5f %.5f %.5f %.5f" % (epoch, epoche_loss, epochd_loss, epochk_loss, epoch_person_loss, epochloss_3d)))
                        f.write('\n')
                        f.close()
                        epoche_loss = 0
                        epochd_loss = 0
                        epochk_loss = 0
                        epochloss_3d = 0
                        epoch_person_loss = 0
                        iter = 0
  
                    if self.count_for_save > mmm:
                        self.sv.request_stop()

                #step += 1

        print('Finish training on %s' % self.model_dir)
