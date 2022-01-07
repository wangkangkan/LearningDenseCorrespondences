'''
    Single-GPU training code
'''

import argparse
import numpy as np
import tensorflow as tf
import importlib
import glob
import os
import sys
import natsort
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import time
from random import shuffle
from write2obj import write_to_obj, read_obj, read_obj_sample


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='../flownet3d/human_1723_3000_stage1/', help='Dataset directory [default: data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')
parser.add_argument('--num_point', type=int, default=1723, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 151]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
LABEL_CHANNEL = 3
EPOCH_CNT = 0
allstep = 200
CLASS_NUM = 21
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL_PATH = './log_train/model.ckpt'

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = 30000

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(train_flag=True):
    if train_flag:
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                pointclouds_pl = {}
                labels_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_POINT, LABEL_CHANNEL])
                pointclouds_pl['pointcloud'] = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_POINT+3000, 104])
                
                idx_1723_tf = None
                is_training_pl = tf.placeholder(dtype=tf.bool)
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                print("--- Get model and loss")
                # Get model and loss
                pred, all_data = \
                    MODEL.get_model(pointclouds_pl, 
                                    idx_1723_tf,
                                    is_training_pl, 
                                    bn_decay=bn_decay)
                loss = MODEL.get_loss(pred, labels_pl, idx_1723_tf)
                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess, coord)
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            if os.path.exists(MODEL_PATH+'.index'):
                saver.restore(sess, MODEL_PATH)
                print('\n!!!!!!!!!!!!!!restore from ', MODEL_PATH)

            ops = {'is_training_pl': is_training_pl,
                'pl': pointclouds_pl['pointcloud'],
                'labels': labels_pl,
                'pred': pred,
                'loss': loss,
                'learning_rate': learning_rate, 
                'train_op': train_op,
                'step': batch}

            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                loss_average, true_epoch = train_one_epoch(sess, ops, saver)
                print('loss_average', loss_average, loss_average/1723)

            coord.request_stop()
            coord.join(threads)
   
def train_one_epoch(sess, ops, saver):
    is_training = False

    down_file_name = '0.txt' 
    down = np.loadtxt(down_file_name).astype(np.int)
    idx_1723 = down[:, 1]

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT+3000, 104))
    batch_label = np.ones((BATCH_SIZE, NUM_POINT, LABEL_CHANNEL))
    global_count2 = 0
    STAGE_1 = "../hks_part3_faust/cape/cape_test/stage1/"

    SAVE_DIR = "../hks_part3_faust/cape/cape_test/stage2/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    allfilename_feat2 = natsort.natsorted(glob.glob("{}*.txt".format(STAGE_1)))
    print(len(allfilename_feat2))

    num_batches = len(allfilename_feat2) // BATCH_SIZE

    loss_sum = 0
    loss_batch = 0
    for batch_idx in range(0, num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        xyz2list = allfilename_feat2[start_idx:end_idx]

        for i, line in enumerate(xyz2list):
            xyz2name = line.split('/')[-1]
            xyz2name = xyz2name.split('.')[0]
            batch_data[i, :NUM_POINT, 0:3] = np.loadtxt(STAGE_1 + "{}.txt".format(xyz2name))

            batch_data[i, NUM_POINT:, 0:3] = np.loadtxt("../hks_part3_faust/cape/cape_test/input_pc/{}.txt".format(xyz2name))
            vert_6890 = np.loadtxt('../hks_part3_faust/cape/cape_test/label/{}.txt'.format(xyz2name))
            batch_label[i, :, :3] = vert_6890[idx_1723, ...]

            global_count2 += 1
        batch_label[:, :, :3] -= batch_data[:, 0:NUM_POINT, 0:3]

        feed_dict = {
                     ops['is_training_pl']: is_training,
                     ops['pl']: batch_data,
                     ops['labels']: batch_label,
                     }
        timestart = time.time()
        step, loss_val, learning_rate_pl, pred_val = sess.run([ops['step'], \
                                                                            ops['loss'], \
                                                                            ops['learning_rate'], \
                                                                            ops['pred']], \
                                                                            feed_dict)
        timeend = time.time()
        loss_sum += loss_val
        loss_batch += loss_val
        print('time::%.3f, lr:%.10f, step:%d, loss: %.3f' % \
            ((timeend-timestart), learning_rate_pl, step, loss_val))
    return loss_batch / num_batches, step//num_batches



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train_flag = True
    train(train_flag)
    LOG_FOUT.close()
