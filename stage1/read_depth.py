import tensorflow as tf 
import numpy as np
import os
import cv2
from write2obj import read_obj_only_vert

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
    """

    def __init__(self, sess=None):
        # Create a single Session to run all image coding calls.
        if sess is None:
            self._sess = tf.Session()
        else:
            self._sess = sess

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

        self._decode_png_data16 = tf.placeholder(dtype=tf.string)
        self._decode_png16 = tf.image.decode_png(
            self._decode_png_data16, dtype=tf.uint16)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png16(self, image_data):
        image = self._sess.run(
            self._decode_png16, feed_dict={
                self._decode_png_data16: image_data
            })
        assert len(image.shape) == 3
        return image


def gene_label(alllabelfiles, num_sample=3000):

    
    allimagefiles = alllabelfiles.replace('testmodelcor', 'testdepth')
    # allimagefiles = alllabelfiles.replace('modelcor', 'depth')
    allimagefiles = allimagefiles.replace('_cor', '')

    depthimage = cv2.imread(allimagefiles,-1)
    # print(type(depthimage))
    # exit()

    height, width = depthimage.shape[:2]
    
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002

    imx = np.tile(np.arange(width), (height, 1))
    imy = np.tile(np.reshape(np.arange(height), [-1, 1]), (1, width))
    imxlist = np.reshape(imx, [-1])
    imylist = np.reshape(imy, [-1])
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 1000.0
    imxlist = (imxlist - cx_d) / fx_d * depthlist
    imylist = (imylist - cy_d) / fy_d * depthlist


    ##   depth ##################################
    pidx = np.reshape(np.where(depthlist > 0), [-1])
    px = np.reshape(imxlist[pidx], [-1, 1])
    py = np.reshape(imylist[pidx], [-1, 1])
    pz = np.reshape(depthlist[pidx], [-1, 1])
    pointset = np.concatenate([px, py, pz], 1)

    # num_sample = sample_num

    pnum = pointset.shape[0]
    if pnum >= 1:
        if (pnum == num_sample):
            sidx = range(num_sample)
        elif (pnum > num_sample):
            sidx = np.random.choice(pnum, num_sample)
        else:
            sample = np.random.choice(pnum, num_sample - pnum)
            sidx = np.concatenate([range(pnum), sample], 0)

        samplepointset = pointset[sidx, :]
    else:
        samplepointset = np.zeros((num_sample, 3))



    # cor = join('../hm36cor/', alllabelfiles[i])

    ##   label ###################################
    labelimage = cv2.imread(alllabelfiles,-1)
    labellist = np.reshape(np.array(labelimage, dtype=np.int32), [-1]) - 1
    select = np.reshape(np.where(labellist > -1), [-1])
    x = np.reshape(imxlist[select], [-1])
    y = np.reshape(imylist[select], [-1])
    z = np.reshape(depthlist[select], [-1])
    vidx = np.reshape(labellist[select], [-1])
    
    modelx = np.zeros(6890)
    modely = np.zeros(6890)
    modelz = np.zeros(6890)
    modelvis = np.zeros(6890)
    modelx[vidx] = x
    modely[vidx] = y
    modelz[vidx] = z
    nvis = np.array(np.greater(vidx, -1), np.float32)
    modelvis[vidx] = nvis
    
    modelx = np.reshape(modelx, [-1, 1])
    modely = np.reshape(modely, [-1, 1])
    modelz = np.reshape(modelz, [-1, 1])
    modelvis = np.reshape(modelvis, [-1, 1])
    modelxyz = np.concatenate([modelx, modely, modelz], 1)
    # modelxyz = modelxyz - np.tile(pointcenter,(modelxyz.shape[0],1))
    lable = np.concatenate([modelxyz, modelvis], 1)
    lable = lable.reshape([1, 6890, 4])

    return samplepointset, lable

def sample_from_depth(image_path, coder=None, num_sample=3000):
    depthimage = cv2.imread(image_path,-1)

    height, width = depthimage.shape[:2]

    # labelimage = coder.decode_png16(label_imagedata)

    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    imx = np.tile(np.arange(width), (height, 1))
    imy = np.tile(np.reshape(np.arange(height), [-1, 1]), (1, width))
    imxlist = np.reshape(imx, [-1])
    imylist = np.reshape(imy, [-1])
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 100.0
    imxlist = (imxlist - cx_d) / fx_d * depthlist
    imylist = (imylist - cy_d) / fy_d * depthlist

    pidx = np.reshape(np.where(depthlist > 0), [-1])
    px = np.reshape(imxlist[pidx], [-1, 1])
    py = np.reshape(imylist[pidx], [-1, 1])
    pz = np.reshape(depthlist[pidx], [-1, 1])
    pointset = np.concatenate([px, py, pz], 1)


    # num_sample = 2500  # 6890
    pnum = pointset.shape[0]
    if pnum >= 1:
        if (pnum == num_sample):
            sidx = range(num_sample)
        elif (pnum > num_sample):
            sidx = np.random.choice(pnum, num_sample)
        else:
            sample = np.random.choice(pnum, num_sample - pnum)
            sidx = np.concatenate([range(pnum), sample], 0)

        samplepointset = pointset[sidx, :]
    else:
        samplepointset = np.zeros((num_sample, 3))

    return samplepointset # (6890, 3)


if __name__ == "__main__":
    import os
    import glob
    import natsort
    from tqdm import tqdm

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    vert = sample_from_depth('00032_shortlong_hips_000037_0.png')
    np.savetxt('pc.txt', vert)