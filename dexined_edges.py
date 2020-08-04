"""
Load in the DexiNed edge detector model from the frozen graph file and use
it to predict edges on abitrary images.

TODO: You must have the DexiNed frozen graph file saved in:
    ./checkpoints/dexined_frozen_graph_{v1,v2}.pbtxt

Download them from:
gs://ds-osama/postprocess/dexined/dexined_frozen_graph_{v1,v2}.pbtxt
"""

###############################################################################
# TODO: Ensure the model directory pointed to exists, see README for details.
DEXINED_MODEL_PATH = "./checkpoints/dexined_keras_model"

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

import numpy as np
import skimage.transform

# %tensorflow_version 1.1x
import tensorflow as tf

# Ignore deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

from custom_logging import debug

###############################################################################

# Load the model in.
debug(f"Loading model from {DEXINED_MODEL_PATH}")
model = tf.saved_model.load_v2(DEXINED_MODEL_PATH)
# rgb_mean = [103.939, 116.779, 123.68]

session = tf.keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

def normalize(img):
    """
    This is a typical image normalization function to normalize the image
    to values in the range [0, 255].
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    Args:
        :img: - a numpy array, either grayscale or RGB

    Returns:
        :img_normed: - a uint8 numpy array, with all values in range [0, 255]
    """
    img = np.float32(img)
    epsilon = 1e-12
    v_min = np.min(img)
    v_max = np.max(img) - np.min(img) + epsilon

    img_normed = (img - v_min) / v_max
    img_normed = np.uint8(img_normed * 255.0)
    return img_normed


def get_dexined_edges(img):
    """
    Helper function to run the model on an arbitrary image.

    Args:
        :img: - an RGB image as a numpy array

    Returns:
        :edgemap: - a grayscale image, of the same height/width
            as the input img.
    """
    h, w, _ = img.shape
    img = img.astype(np.float32)

    # Image must be float32 and range from 0.0 to 255.0
    if np.max(img) <= 1.0:
        img *= 255.0

    img = skimage.transform.resize(img, (512, 512),
            preserve_range=True)

    img_batch = img.reshape((1, 512, 512, -1))
    img_batch = tf.constant(img_batch)
    with session.as_default():
        edgemap = model(img_batch).eval().squeeze()
        edgemap[edgemap <= 0] = 0.0
        # Normalize to values in range [0, 255]
        edgemap = np.uint8(edgemap * 255)
        edgemap = skimage.transform.resize(edgemap, (h, w),
                preserve_range=True)

    return edgemap


if __name__ == "__main__":
    import numpy as np
    from imageio import imread, imwrite

    src_fpath = "figures/living_room.jpg"
    dst_fpath = "figures/living_room_edges.png"

    # Read in an RGB image as a numpy array.
    img = imread(src_fpath, pilmode="RGB")
    img = np.asarray(img)

    # Get the edgemap, which is a grayscale uint8 numpy array.
    edgemap = get_dexined_edges(img)
    imwrite(dst_fpath, edgemap)

    print(f"Saved edgemap of {src_fpath} to {dst_fpath}")
