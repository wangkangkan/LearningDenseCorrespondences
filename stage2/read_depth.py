import tensorflow as tf 
import numpy as np
import os

def gene_label(alllabelfiles, num_sample=3000):

    import cv2
    
    # allimagefiles = alllabelfiles.replace('testmodelcor', 'testdepth')
    allimagefiles = alllabelfiles.replace('modelcor', 'depth')
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



def sample_from_depth(image_path, coder, num_sample=3000):
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        depth_imagedata = f.read()

    depthimage = coder.decode_png16(depth_imagedata)
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
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 1000.0
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
    pass