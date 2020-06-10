"""
Load in the DexiNed edge detector model from the frozen graph file and use
it to predict edges on abitrary images.

TODO: You must have the DexiNed frozen graph file saved in:
    ./checkpoints/dexined_frozen_graph.pbtxt

You can either:
(a) run `python ExportNetwork.py` to generate it.

(b) Download it from:
gs://ds-osama/postprocess/dexined/dexined_frozen_graph.pbtxt

The main() function here just serves as a test.
"""

###############################################################################
# TODO: Change to your own path to the tensorflow models repo
DEXINED_MODEL_PATH = "checkpoints/dexined_frozen_inference_graph.pbtxt"

###############################################################################

# Ignore the annoying warnings that come up, there are many of them.
import warnings
import os
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# TODO: Uncomment line to disable GPU usage. By default, GPU usage is enabled.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    # This is the name we gave to the network architecture input with a ":0"
    # at the end to indicate we're providing the first such tensor as input.
    INPUT_TENSOR_NAME = 'ImageTensor:0'

    # These are the names from output_node_names, but with a ":0" at the end
    # to indicate we want the first tensor of that name.
    OUTPUT_TENSOR_NAME = ['output_0:0', 'output_1:0', 'output_2:0',
        'output_3:0', 'output_4:0', 'output_5:0', 'output_6:0']

    # This is the name of the frozen graph file. Even if this file is in a tar
    # or zip archive, we want to make sure to read only this file to build the
    # TensorFlow graph.
    FROZEN_GRAPH_NAME = 'dexined_frozen_graph.pbtxt'

    # Ensure these are the same as from DexinedNetwork.
    TARGET_H = TARGET_W = 512
    N_CHANNELS = 3

    def __init__(self, model_init_path):
        """
        Loads the pretrained DexiNed model using the frozen graph file.

        Args:
            :model_init_path: - A path to either a tar archive with the frozen
                graph file inside, or the path to the frozen graph file itself.
        """
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
            raise RuntimeError('Cannot find frozen inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.session = tf.compat.v1.Session(graph=self.graph)


    def run(self, img):
        """
        Produce the output edgemap.

        Args:
            :img: - an RGB image as a numpy array

        Returns:
            :avg_edgemap: - a grayscale image, of the same height/width
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

        # This model produces 7 edge maps total, each of shape (1 x H x W x 1)
        batch_edge_maps = self.session.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: img_batched})

        # Here we squeeze each map, so they each have dimensions (H x W)
        edge_maps = [em.squeeze() for em in batch_edge_maps]

        # We average these edge maps together to obtain the best edge detection
        # results. This is the final edge map, with dimensions (H x W)
        avg_edgemap = np.mean(np.array(edge_maps), axis=0)

        # Resize the edgemap to the same size as the input image.
        avg_edgemap = skimage.transform.resize(avg_edgemap, src_dimensions)

        # Finally, invert the colors so edges are 0, non-edges are 255, with
        # all values in between being weak/strong edges.
        avg_edgemap = (255.0 * (1.0 - avg_edgemap)).astype(np.uint8)

        return avg_edgemap

###############################################################################

print(f"Loading model {DEXINED_MODEL_PATH}")
model = DexinedModel(DEXINED_MODEL_PATH)
print(f"Finished loading model.")


def get_dexined_edges(img):
    """
    Helper function to run the model on an arbitrary image.

    Args:
        :img: - an RGB image as a numpy array

    Returns:
        :avg_edgemap: - a grayscale image, of the same height/width
            as the input img.
    """
    return model.run(img)


def main():
    img_uri = "data/start.png"
    # Read image as RGB image.
    img = imread(img_uri, pilmode="RGB")
    img = np.asarray(img)

    em = get_dexined_edges(img)
    imwrite("g0.png", img)
    imwrite("g1.png", em)

if __name__ == "__main__":
    main()