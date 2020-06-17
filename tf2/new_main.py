#!/usr/bin/python3.7 python
"""

"""
__author__ = "Xavier Soria Poma"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"
__homepage__="www.cvc.uab.cat/people/xsoria"
__credits__=["tensorflow_tutorial"]
__copyright__   = "Copyright 2020, CIMI"

# Ignore the annoying warnings that come up, there are many of them.
import warnings
import os
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import argparse
import tensorflow as tf

from runner import run_DexiNed


# Testing settings
DATASET_NAME= ['BIPED','BSDS','BSDS300','CID','DCD','MULTICUE',
    'PASCAL','NYUD','CLASSIC']

TEST_DATA = 'BIPED'
TRAIN_DATA = 'BIPED'

# Training settings

parser = argparse.ArgumentParser(description='Edge detection parameters for feeding the model')
parser.add_argument("--output_dir", default='results', help="where to put output files")
parser.add_argument("--checkpoint_dir", default='checkpoints', help="directory with checkpoint to resume training from or use for testing")

parser.add_argument('--model_name', default='DexiNed', choices=['DexiNed'])
parser.add_argument('--continue_training', default=False, type=bool)
parser.add_argument("--max_epochs", type=int,default=2, help="number of training epochs")#24
parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=10, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=10, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--test_bs", type=int, default=1, help="number of images in test batch")
parser.add_argument("--batch_normalization", type=bool, default=True, help=" use batch norm")
parser.add_argument("--image_height", type=int, default=400, help="scale images to this size before cropping to 256x256")
parser.add_argument("--image_width", type=int, default=400, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_img", type=bool, default=False,
                    help="4Training: True crop image, False resize image")
parser.add_argument("--test_img_height", type=int, default=720,
                    help="network input height size")
parser.add_argument("--test_img_width", type=int, default=720,
                    help="network input height size")

parser.add_argument("--lr", type=float, default=0.0002, help=" learning rate for adam 1e-4")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--rgbn_mean", type=float, default=[103.939,116.779,123.68, 137.86], help="pixels mean")

arg = parser.parse_args()

def main(args):
    model = run_DexiNed(epochs=args.max_epochs)

    if args.model_state=='train':
        model.train()
    elif args.model_state =='test':
        pass
        # model.test()
    else:
        raise NotImplementedError('Sorry you just can test or train the model, please set in ' +
            'args.model_state=')

if __name__=='__main__':
    main(args=arg)
