import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.lesion_inpainting import GANLesionInpainting


def main(mode=None):
    """
    starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = GANLesionInpainting(config)

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.predict()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1], help='1: inpaint model')
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--run-id', type=int, default=1)
    parser.add_argument('--best-hparams')
    parser.add_argument('--native', action="store_true")
    parser.add_argument('--model-3D', action="store_true")

    # test mode
    if mode == 2:
        parser.add_argument('--output', type=str, help='path to the output directory')
    #     parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    #     parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')

    if mode == 3:
        parser.add_argument('--input-image', type=str)
        parser.add_argument('--input-lesion', type=str)
        parser.add_argument('--brain-mask', type=str)
        parser.add_argument('--tissue-mask', type=str)
        parser.add_argument('--xfm', type=str)
        parser.add_argument('--outdir', type=str)


    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    config.iteration = args.iteration
    config.run_id = args.run_id
    config.model_3D = args.model_3D
    if config.model_3D:
        config.z_size = 8
    else:
        config.z_size = 1

    if args.best_hparams:
        config.run_id = args.best_hparams

    config.NATIVE = False
    if not args.native:
        config.XFM_FLIST = None
    else:
        config.NATIVE = True

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        # if args.input is not None:
        #     config.TEST_FLIST = args.input
        #
        # if args.mask is not None:
        #     config.TEST_MASK_FLIST = args.mask
        #
        # if args.edge is not None:
        #     config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

        config.input_image = args.input_image
        config.input_lesion = args.input_lesion
        config.brain_mask = args.brain_mask
        config.tissue_mask  = args.tissue_mask
        config.xfm = args.xfm
        config.outdir = args.outdir

    return config


if __name__ == "__main__":
    main()
