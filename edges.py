"""
TODO: Download the following to your local machine and change the paths accordingly:
    gs://ds-osama/postprocess/deploy.prototxt
    gs://ds-osama/postprocess/hed_pretrained_bsds.caffemodel
    gs://ds-osama/postprocess/frozen_inference_graph.pb

"""

###############################################################################
# TODO: Change to your own path to the tensorflow models repo
DEXINED_MODEL_PATH = "checkpoints/frozen_graph.pbtxt"

###############################################################################

# Ignore the annoying warnings that come up, there are many of them.
import warnings
import os
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


# TODO: Comment out line below to enable GPU usage. By default, No GPU is used.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###############################################################################

import os
import tarfile

import numpy as np
import matplotlib

import skimage
import skimage.transform
from imageio import imread, imwrite

# %tensorflow_version 1.x
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Logging - https://github.com/ziploc1010/thd-visual-ai/tree/master/ds_toolbox
from ds_toolbox.ds_logging import set_log_level, add_log_file
from ds_toolbox.ds_logging import debug, info, warn, error
set_log_level("debug")

###############################################################################

# These are to enable Model to run on GPU
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# Ignore additional warnings and logs from TensorFlow.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set SEED for numpy
SEED = 10
np.random.seed(SEED)

###############################################################################

class DexinedModel(object):
    """Class to load DexiNed model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'output_6:0'
    INPUT_SIZE = 512
    FROZEN_GRAPH_NAME = 'frozen_graph.pbtxt'

    TARGET_H = TARGET_W = 512
    N_CHANNELS = 3

    def __init__(self, model_init_path):
        """Creates and loads pretrained DexiNed model."""
        self.graph = tf.compat.v1.Graph()
        graph_def = None

        if 'tar.gz' in model_init_path:
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(model_init_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                    break

            tar_file.close()

        elif self.FROZEN_GRAPH_NAME in model_init_path:
            # Extract model from a prozen_inference_graph.pb file
            with open(model_init_path, 'rb') as rf:
                graph_def = tf.compat.v1.GraphDef.FromString(rf.read())

        else:
            raise ValueError(f"Unexpected file type: {model_init_path}")

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)


    def run(self, img):
        """
        Produce the output edgemap.

        Args:
            :img: - an RGB image as a numpy array

        Returns:
            :final_edgemap: - a 1-channel image, of the same height/width
                as the input img.
        """

        # If the image is not a float32 image, convert it.
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Get dimensions of the original image, store it to resize the edgemaps
        # later.
        src_h, src_w, _ = img.shape
        src_dimensions = (src_h, src_w)


        # Remove mean RGB value from image.
        R = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        B = np.mean(img[:, :, 2])

        img[:, :, 0] -= R
        img[:, :, 1] -= G
        img[:, :, 2] -= B

        # Resize image so it can be passed through the network.
        target_dimensions = (self.TARGET_H, self.TARGET_W)
        img = skimage.transform.resize(img, target_dimensions)

        # Turn this single image into a batch, the shape will be (1 x H x W x 3)
        img_batched = img.reshape((1, self.TARGET_H, self.TARGET_W,
            self.N_CHANNELS))

        # Feed into the network to get back the edgemaps
        # edge_maps = self.session.run(self.predictions, feed_dict={self.images: img_batched})

        # Average the edgemaps and resize into the original image dimensions
        # final_edgemap = self.get_single_edgemap(edge_maps, src_dimensions)

        batch_edge_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: img_batched})

        edge_map = batch_edge_map[0]

        # TODO: Resize to original dims
        final_edgemap = skimage.transform.resize(edge_map, src_dimensions)

        return np.asarray(final_edgemap)

###############################################################################

debug(f"Loading model {DEXINED_MODEL_PATH}")
model = DexinedModel(DEXINED_MODEL_PATH)
debug(f"Finished loading model.")

if __name__ == "__main__":
    img_uri = "data/stairs.jpg"
    # Read image as RGB image.
    img = imread(img_uri, pilmode="RGB")
    img = np.asarray(img)

    final_edgemap = model.run(img)
    imwrite("g0.png", img)
    imwrite("g1.png", final_edgemap)