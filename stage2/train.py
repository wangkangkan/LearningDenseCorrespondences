import argparse
import numpy as np
import tensorflow as tf
import importlib
# import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='../flownet3d/human_1723_3000_stage2_new/', help='Dataset directory [default: data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--log_dir', default='log_train_epoch1', help='Log dir [default: log_train]')
parser.add_argument('--num_point', type=int, default=1723, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 151]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

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


MODEL_PATH = './log_train_epoch/model.ckpt'

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
# os.system('cp %s %s' % ('flying_things_dataset.py', LOG_DIR)) # bkp of dataset file
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = 300000

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

def parse_example(example_serialized, pointnum=6890, pointnum2=3000, k=300):
    """Parses an Example proto."""
    feature_map = {
        'frame1/x': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame1/y': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame1/z': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame2/x': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'frame2/y': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'frame2/z': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'flow/x' : tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'flow/y' : tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'flow/z' : tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    frame1_x = tf.cast(features['frame1/x'], dtype=tf.float32)
    frame1_y = tf.cast(features['frame1/y'], dtype=tf.float32)
    frame1_z = tf.cast(features['frame1/z'], dtype=tf.float32)
    frame1_x = tf.reshape(frame1_x, [pointnum, 1])
    frame1_y = tf.reshape(frame1_y, [pointnum, 1])
    frame1_z = tf.reshape(frame1_z, [pointnum, 1])
    frame1 = tf.concat([frame1_x, frame1_y, frame1_z], axis=1)

    feat1 = tf.zeros([pointnum, 101])
    feat2 = tf.zeros([pointnum2, 101])

    frame2_x = tf.cast(features['frame2/x'], dtype=tf.float32)
    frame2_y = tf.cast(features['frame2/y'], dtype=tf.float32)
    frame2_z = tf.cast(features['frame2/z'], dtype=tf.float32)
    frame2_x = tf.reshape(frame2_x, [pointnum2, 1])
    frame2_y = tf.reshape(frame2_y, [pointnum2, 1])
    frame2_z = tf.reshape(frame2_z, [pointnum2, 1])
    frame2 = tf.concat([frame2_x, frame2_y, frame2_z], axis=1)

    frame = tf.concat([frame1, frame2], axis=0)
    color = tf.concat([feat1, feat2], axis=0)
    frame = tf.concat([frame, color], axis=1)

    flow_x = tf.cast(features['flow/x'], dtype=tf.float32)
    flow_y = tf.cast(features['flow/y'], dtype=tf.float32)
    flow_z = tf.cast(features['flow/z'], dtype=tf.float32)
    flow_x = tf.reshape(flow_x, [pointnum, 1])
    flow_y = tf.reshape(flow_y, [pointnum, 1])
    flow_z = tf.reshape(flow_z, [pointnum, 1])
    flow = tf.concat([flow_x, flow_y, flow_z], axis=1)

    return frame, flow#, vector, value#, frame2flow

def get_batch(batch_size, fqueue):

    with tf.name_scope(None, 'read_data', [fqueue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(fqueue)
        frame, flow = parse_example(example_serialized, pointnum=NUM_POINT)
        min_after_dequeue = 160
        num_threads = 4
        capacity = min_after_dequeue + 3 * batch_size

        pack_these = [frame, flow]
        pack_name = ['frame', 'flow']
        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')

        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

def train(train_flag=True):
    if train_flag:
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                filelist = os.listdir(DATA)
                data_dirs = filelist
                all_files = []
                for data_dir in data_dirs:
                    all_files.append(DATA + data_dir)
                pointclouds_pl = {}
                do_shuffle = True
                fqueue = tf.train.string_input_producer(all_files, shuffle=do_shuffle, name="input")
                batch_dict = get_batch(BATCH_SIZE, fqueue)
                pointclouds_pl['pointcloud'], labels_pl = batch_dict['frame'], batch_dict['flow']

                is_training_pl = tf.placeholder(dtype=tf.bool)

                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                print("--- Get model and loss")
                # Get model and loss
                pred, all_data = \
                    MODEL.get_model(pointclouds_pl, 
                                    is_training=is_training_pl, 
                                    bn_decay=bn_decay)
                loss = MODEL.get_loss(pred, labels_pl)

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
                if epoch % 1 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)
            
                    with open('./log_train/loss.txt', 'a+') as f:
                        if epoch == 0:
                            f.write('===1723 concat=============================================\n')
                        f.write('epoch:%d, loss: %06f \n' % (true_epoch, loss_average/1723))

            coord.request_stop()
            coord.join(threads)
   
def train_one_epoch(sess, ops, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    num_batches = 24800

    loss_sum = 0
    loss_batch = 0
    for batch_idx in range(0, num_batches):
        timestart = time.time()
        feed_dict = {
                     ops['is_training_pl']: is_training,
                     }
        step, _,  loss_val, learning_rate_pl, pred_val = sess.run([ops['step'], \
                                                                            ops['train_op'], \
                                                                            ops['loss'], \
                                                                            ops['learning_rate'], \
                                                                            ops['pred']], \
                                                                            feed_dict)
        timeend = time.time()
        loss_sum += loss_val
        loss_batch += loss_val
        print('time::%.3f, lr:%.10f, step:%d, loss: %.3f' % \
            ((timeend-timestart), learning_rate_pl, step, loss_val))

    return loss_batch / num_batches, step // num_batches 

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train_flag = True
    train(train_flag)
    LOG_FOUT.close()
