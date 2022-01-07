#!/usr/bin/python
import numpy as np
import os
import re
import sys
from scipy.io import loadmat
from read_depth import ImageCoder, sample_from_depth

import argparse

import tensorflow as tf

import warnings
warnings.filterwarnings('error')

def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to_example(frame1, frame2, flow, frm1_point=6890, frm2_point=3000):
    """Build an Example proto for an image example.
    Args:
      frame1: 
      frame2:
      flow: label

    Returns:
      Example proto
    """

    frame1 = np.reshape(frame1, [frm1_point, 3])
    frame2 = np.reshape(frame2, [frm2_point, 3])
    flow = np.reshape(flow, [frm1_point, 3])

    feat_dict = {
        'frame1/x': float_feature(frame1[:, 0].astype(np.float)),
        'frame1/y': float_feature(frame1[:, 1].astype(np.float)),
        'frame1/z': float_feature(frame1[:, 2].astype(np.float)),
        'frame2/x': float_feature(frame2[:, 0].astype(np.float)),
        'frame2/y': float_feature(frame2[:, 1].astype(np.float)),
        'frame2/z': float_feature(frame2[:, 2].astype(np.float)),
        'flow/x': float_feature(flow[:, 0].astype(np.float)),
        'flow/y': float_feature(flow[:, 1].astype(np.float)),
        'flow/z': float_feature(flow[:, 2].astype(np.float)),
    }
    

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example


def proc_one_scene_human(allmodelfiles, frame2_hks,flowfile, output_dir):
    train_or_test = 'train'
    out_path = os.path.join(output_dir, train_or_test + "_dfaust_" + '%03d.tfrecord')

    down_file_name = '0.txt' 
    down = np.loadtxt(down_file_name).astype(np.int)
    idx_1723 = down[:, 1]
    frame1 = np.loadtxt('../flownet3d/data_preprocessing/template_6890.txt')
    fidx = 0
    seqnum = len(allmodelfiles)
    print(seqnum)
    i = 0
    tnum_shards = 1000
    while i < seqnum:
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < seqnum and j < tnum_shards:
                xyz2dir = allmodelfiles[i]
                print(allmodelfiles[i])
                xyz2name = xyz2dir.split('male_')[-1]
                xyz2name = xyz2name.replace('_-1.png', '.txt')

                
                print('Converting image %d/%d' % (i, seqnum))
                frame2 = sample_from_depth(xyz2dir, None, num_sample=3000)
                frame2_true = np.loadtxt('../DFAUS/male/dfaust_male_transed_txt/'+xyz2name)
                flow = frame2_true - frame1

                example = convert_to_example(frame1, frame2, flow, frm2_point=3000) 
                writer.write(example.SerializeToString())

                j += 1
                i += 1


        fidx += 1

def read_file_list(filelist):
    """
    Scan the image file and get the image paths and labels
    """
    with open(filelist) as f:
        lines = f.readlines()
        files = []  
        for l in lines:
            items = l.split()
            # print(items)
            # exit()
            files.append(items[0])
            #self.imagefiles.append(l)

        # store total number of data
    filenum = len(files)
    print("Training sample number: %d" % (filenum))
    return files


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    allmodelfiles = read_file_list('../DFAUS/male/rendermale/depth/filenamelist.txt') # train_8_real_3k
    OUTPUT_DIR = "./"
    proc_one_scene_human(allmodelfiles, None, None, OUTPUT_DIR)

