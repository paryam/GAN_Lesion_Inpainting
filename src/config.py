import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.full_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MASK': 2,                      # 1: t2-lesions, 2: external, 3: random, 4: all
    'NMS': 1,                       # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'T2W_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/t2w_images/t2w_images_final.txt",
    'LESION_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/lesions_final.txt",
    'BRAIN_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/brain_masks_final.txt",

    'TRAIN_MASK_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/masks_train.flist",
    'VAL_MASK_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/masks_val.flist",
    'TEST_MASK_FLIST': "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/edge-connect/datasets/masks_test.flist",

    'LR': 0.0001,                   # learning rate
    'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 32,               # input batch size for training
    'VAL_BATCH_SIZE': 32,           # input batch size for validation
    'INPUT_SIZE': 256,              # input image size for training 0 for original size
    'SIGMA': 2,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e7,               # maximum number of iterations to train the model
    'MIN_MASK_SIZE': 300,

    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 1,         # style loss weight
    'CONTENT_LOSS_WEIGHT': 1,       # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.05, # adversarial loss weight

    'GAN_LOSS': 'nsgan',            # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,             # fake images pool size

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)

    'MINC_DIR': "/scratch/02/paryam/conda-envs/miniconda3/envs/pytorch/bin/",
}
