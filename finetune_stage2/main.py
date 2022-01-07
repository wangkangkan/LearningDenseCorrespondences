import tensorflow as tf

from config_our_3dpoint import get_config, prepare_dirs, save_config
from data_loader_our_3dpoint import DataLoader
from trainer_our_3dpoint_ournetwork_feed import HMRTrainer

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)

        smpl_loader = data_loader.get_smpl_loader()

    trainer = HMRTrainer(config, None, smpl_loader)
    save_config(config)
    trainer.train_feed()


if __name__ == '__main__':
    config = get_config()
    main(config)
