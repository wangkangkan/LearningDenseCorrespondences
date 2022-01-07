from data_loader_our_3dpoint import num_examples

from ops import keypoint3D_loss, compute_3d_loss_our
from models import Discriminator_separable_rotations, get_encoder_fn_separate

from tf_smpl.batch_lbs import batch_rodrigues
from tf_smpl.batch_smpl_our import SMPL
from tf_smpl.projection import batch_orth_proj_idrot

from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np
from lib.decodeModel import DecodeModel, read_sp_matrix

from os.path import join, dirname, split
import glob
import random
from write2obj import write_to_obj
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
        self.save_dir = "/test/hks_part3_faust/cape/cape_test/"
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

        #changed to RT
        self.proj_fn = batch_orth_proj_idrot

        #including R
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Data
        num_images = 1250
        num_mocap = num_examples(config.mocap_datasets)

        self.num_itr_per_epoch = int(num_images / self.batch_size)
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        self.pointset_loader = tf.placeholder(tf.float32, shape=[self.batch_size, 1723, 3])

        self.count_for_save = 0

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
        # if ('resnet' in self.model_type) and (self.pretrained_model_path is
        #                                       not None):
        #     # Check is model_dir is empty
        #     import os
        #     if os.listdir(self.model_dir) == []:
        #         return True

        return True

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        #mean[0] = np.pi
        mean_path = './meanpara_male.txt'
        mean_vals = np.loadtxt(mean_path, dtype=np.float32)

        mean_pose = mean_vals[:72]
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals[72:]

        #This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, :self.total_params-3] = np.hstack((mean_pose, mean_shape))

        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        #self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean

    def l1_loss(self, pred, label):
        return tf.reduce_mean(tf.reduce_sum(label[:, :, 3] * tf.sqrt(tf.reduce_sum(tf.abs(pred-label[:, :, :3]), axis=-1)), axis=1), axis=0)

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

        inputpoints = self.pointset_loader#self.kp_loader[:,:,0:3]
        # self.img_feat, self.E_var= pointnet2.extract_globalfeature(self.pointset_loader, None, tf.cast(True, tf.bool))
        self.img_feat = tf.reshape(self.pointset_loader, [self.batch_size, -1])
        self.E_var = []

        loss_kps = []
        loss_personal = []
        loss_3d_params = []
        # For discriminator
        fake_rotations, fake_shapes = [], []
        # Start loop
        # 85D
        pred_pose, pred_shape, pred_cam = self.load_mean_param1()
        self.E_var.extend([self.init_pose, self.init_shape, self.init_cam])

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
            state = tf.concat([self.img_feat, pred_pose, pred_shape, pred_cam], 1)

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
            verts, pred_Rs = self.smpl(pred_shape, pred_rotmat)

            pred_kp = verts + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating
            # pred_kp_persnoal = verts_personal + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating
            pred_kp_1723 = tf.gather(pred_kp, self.idx_1723_tf, axis=1)
            # --- Compute losses:
            loss_kps.append(self.e_loss_weight * self.keypoint_loss(
                self.pointset_loader, pred_kp_1723))

            self.all_theta_prev.append(tf.concat([pred_pose, pred_shape, pred_cam], axis=1))
            self.all_pred_kps.append(pred_kp)


            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])

            # Save pred_rotations for Discriminator
            fake_rotations.append(pred_Rs[:, 1:, :])
            fake_shapes.append(pred_shape)

        if not self.encoder_only:
            self.setup_discriminator(fake_rotations, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_e_loss"):
            # Just the last loss.
            self.e_loss_kp = loss_kps[-1]

            if self.encoder_only:
                self.e_loss = self.e_loss_kp 
            else:
                self.e_loss = self.d_loss_weight * self.e_loss_disc + self.e_loss_kp 

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

    
    def save_result(self, paras, pred_kp, name):
        for i in range(self.batch_size):
            np.savetxt(self.save_dir + "stage3/{}.txt".format(name[i]), pred_kp[i], "%f")
            write_to_obj(self.save_dir + "stage3/{}.obj".format(name[i]), pred_kp[i])

            np.savetxt(self.save_dir + "paras/{}.txt".format(name[i]), paras[i],"%f")
            self.count_for_save += 1

    def train_feed(self):

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
            all_list = sorted(glob.glob(self.save_dir + "stage2/*.txt"))
            mmm = len(all_list)
            print('*'*20, len(all_list))
            inp = np.zeros((self.batch_size, 1723, 3))
            while not self.sv.should_stop():
                random.shuffle(all_list)
                feed_dict={
                    self.pointset_loader: inp
                }
                num_iter = len(all_list) // self.batch_size


                fetch_dict = {
                    "step": self.global_step,
                    "e_loss": self.e_loss,
                    # The meat
                    "e_opt": self.e_opt,
                    "loss_kp": self.e_loss_kp,
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

                ########################### for save results ################
                fetch_dict.update({
                    "pointset": self.pointset_loader,
                    # 'label': self.kp_loader,
                    # 'input_pc': self.input_pc,
                    'paras': self.all_theta_prev[-1],
                    'pred_kp': self.all_pred_kps[-1],
                })

               
                for i in range(num_iter):
                    names = []
                    for j in range(self.batch_size):
                        name = split(all_list[i*self.batch_size + j])[1].rstrip('.txt')
                        names.append(name)
                        inp[j] = np.loadtxt(all_list[i*self.batch_size + j])

                    t0 = time()
                    result = sess.run(fetch_dict, feed_dict={self.pointset_loader : inp})
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
                            "itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f, kp_loss: %.4f, loss_person:%.3f, loss_3d_params: %.4f"
                            % (step, epoch, t1 - t0, e_loss, d_loss, loss_kp, loss_person, loss_3d))

                    # self.save_result(result['paras'], result['pred_kp'], names)

                    epoche_loss = epoche_loss + e_loss
                    epochd_loss = epochd_loss + d_loss
                    epochk_loss = epochk_loss + loss_kp
                    epoch_person_loss += loss_person
                    epochloss_3d = epochloss_3d+ loss_3d
                    iter = iter + 1
                # self.saver.save(sess,'./logs/models/model.ckpt', global_step = step)
                # exit()
                    if step % self.num_itr_per_epoch == 0 and step >= 1:
                        # self.saver.save(sess,'./logs/models/model.ckpt', global_step = int(epoch))
                        f = open('./loss0808.txt', 'a+')
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

                    if self.count_for_save >=mmm:
                        self.sv.request_stop()

        print('Finish training on %s' % self.model_dir)
