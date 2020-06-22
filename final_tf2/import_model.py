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

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img



if __name__ == "__main__":
    model_path = "checkpoints/final_dexined_tf2_model/"
    model = tf.saved_model.load(model_path)
    print(f"Model summary: {dir(model)}")
    print(f"Model loaded")
    print(f"Model type: {type(model)}")
    print(f"RGBN_MEAN: {model.rgbn_mean}")

    img = imread("figures/living_room.jpg").astype(np.float32)
    print(f"{np.min(img)}    {np.max(img)}")

    R = np.mean(img[:, :, 0])
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 2])

    img[:, :, 0] -= R
    img[:, :, 1] -= G
    img[:, :, 2] -= B

    img = skimage.transform.resize(img, (512, 512))
    img_batch = img.reshape((1, 512, 512, -1)).astype(np.float32)
    img_batch = tf.constant(img_batch)

    preds = model(img_batch)
    preds = [tf.sigmoid(i).numpy().squeeze() for i in preds]
    preds = np.array(preds)
    print(f"Preds: {preds.shape}")

    avg = np.mean(preds, axis=0)
    assert (512, 512) == avg.shape
    avg[avg < 0.0] = 0.0
    avg = image_normalization(avg).astype(np.uint8)
    # Invert
    avg = (255 - avg)
    print(f"Avg: {avg.shape} {avg.dtype}")

    imwrite("figures/living_room_alt_edges.png", avg)
