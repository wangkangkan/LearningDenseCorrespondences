# Finetune Stage1 and Finetune Stage2
The code framework refers to [HMR](https://github.com/akanazawa/hmr)

## Prepare 
Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl` into `models`. 

[Download Link](https://pan.baidu.com/s/1N-TsikFeuAqQ8esUqZ2_Xw), extract code：hjun.


## Finetune
`main.py`

1. Get data for the discriminator from [HMR](https://github.com/akanazawa/hmr) and change `DATA_DIR` in `config_our_3dpoint.py`.
2. Change data path in function `train_feed` in `trainer_our_3dpoint_ournetwork_feed.py` to your files path.
3. ```
   # start finetuning 
   python main.py
   ```

### Pre-train model
[Finetune Stage1](https://pan.baidu.com/s/170u8fVzguXh-FCKtL3hjFA ), extract code：0k74.

[Finetune Stage2](https://pan.baidu.com/s/1_t2HXiGZIhNgq81NMaQlvQ ), extract code：jgqb.

If use the pretrain model, you need:

1. Download the pretrain model and change `PRETRAINED_MODEL` and `load_path` in `config_our_3dpoint.py`.
2. Change `encoder_only` to `True` in `config_our_3dpoint.py`. 
3. Change the code of function `train_feed` and comment line446 and line452 for testing.