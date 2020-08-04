import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore")

from imageio import imread, imwrite
import skimage.transform
import numpy as np

import tensorflow as tf
from model import *

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


if __name__ == '__main__':
    model = DexiNed()
    print(f"MODEL: {model.rgb_mean} {type(model.rgb_mean)}")

    img = imread("figures/living_room.jpg").astype(np.float32)
    R, G, B = [103.939,116.779,123.68]

    img[:, :, 0] -= R
    img[:, :, 1] -= G
    img[:, :, 2] -= B

    img = skimage.transform.resize(img, (512, 512))
    img_batch = img.reshape((1, 512, 512, -1)).astype(np.float32)
    img_batch = tf.constant(img_batch)
    model.build(input_shape=img_batch.shape)

    model_path = '../checkpoints/dexined_model_orig.h5'
    model.load_weights(model_path)

    preds = model(img_batch, training=False)
    avg = preds.numpy()
    assert (512, 512) == avg.shape

    avg[avg < 0.0] = 0.0
    avg = normalize(avg)
    # Invert
    avg = (255 - avg)

    out_filepath = "figures/living_room_edges_norm.png"
    imwrite(out_filepath, avg)

    # Save to final checkpoint name
    filepath = "checkpoints/dexined_normalized/"
    print(f"Saving exported model to {filepath}")
    tf.saved_model.save(model, filepath)

    # Save h5
    filepath = "checkpoints/keras_dexined_weights.h5"
    print(f"Saving keras model to {filepath}")
    model.save_weights(filepath)

