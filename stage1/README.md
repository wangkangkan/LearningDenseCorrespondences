# Stage1 and Stage2

### Prepare 
Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl` into `tf_smpl/models`. 

[Download Link](https://pan.baidu.com/s/1N-TsikFeuAqQ8esUqZ2_Xw), extract code：hjun.

### Generate training data

#### Stage1

`gene_tfrecord.py` in `stage1`

For generate training data, we need prepare:
  - Template model, which is `frame1/x y z` in the function `convert_to_example` in the code.
  - Partial pointcloud from depthmap, which is `frame2/x y z` in the function `convert_to_example` in the code.
  - Groundtruth displacement which can be calculated by gt model minus template, which is `flow/x y z` in the function `convert_to_example` in the code.

Run the code 
   ```
   python gene_tfrecord.py
   ```

#### Stage2

`gene_tfrecord.py` in `stage2`

For generate training data, we need prepare:
  - Results generated in Stage1, which is `frame1/x y z` in the function `convert_to_example` in the code.
  - Partial pointcloud from depthmap, which is `frame2/x y z` in the function `convert_to_example` in the code.
  - Groundtruth displacement which can be calculated by gt model minus stage1 results, which is `flow/x y z` in the function `convert_to_example` in the code.

Run the code 
   ```
   python gene_tfrecord.py
   ```



### Train and Test

`train.py`



1. Change `--data` in `train.py` to your .tfrecord files path.
2. Change `MODEL_PATH` for saving model.
3. ```
   # start training 
   python train.py
   ```

`test.py`

[Stage1 pre-train model](https://pan.baidu.com/s/1Y4C1CDB6zkD04aP9yn0XGA )

extract code：l7on

[Stage2 pre-train model](https://pan.baidu.com/s/1idL-UrjM8rfqLl4wOsdWRQ )

extract code：3uif

1. Download the pretrain model and put it in the `MODEL_PATH` in `test.py`
2. change the data path in function `train_one_epoch`
3. ```
   # start testing
   python test.py
   ```