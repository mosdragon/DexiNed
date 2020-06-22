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

if __name__ == "__main__":
    model_path = "checkpoints/final_dexined_tf2_model/"
    model = tf.saved_model.load(model_path)

    img = imread("figures/living_room.jpg")
    print(f"{np.min(img)}    {np.max(img)}")
    img = img.astype(np.float32) / 255.0
    print(f"{np.min(img)}    {np.max(img)}")

    R = np.mean(img[:, :, 0])
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 2])

    # img[:, :, 0] -= R
    # img[:, :, 1] -= G
    # img[:, :, 2] -= B

    img = skimage.transform.resize(img, (512, 512))
    img_batch = img.reshape((1, 512, 512, -1)).astype(np.float32)
    img_batch = tf.constant(img_batch)

    # model.build(input_shape=img_batch.shape)
    preds = model(img_batch)
    preds = tf.concat(preds, 3)  # B x W x H x 7
    print(f"Preds: {preds.shape}")

    preds = preds.numpy().squeeze()
    avg = np.mean(preds, axis=-1)
    avg[avg < 0.0] = 0.0
    print(f"Avg: {avg.shape}")

    imwrite("figures/living_room_edges.png", avg)
