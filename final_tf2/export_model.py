import warnings
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore")

from imageio import imread, imwrite
import skimage.transform
import numpy as np

import cv2
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

if __name__ == '__main__':
    RGBN_MEAN = [103.939,116.779,123.68, 137.86]
    model = DexiNed(rgb_mean=RGBN_MEAN)

    img = imread("figures/living_room.jpg")
    print(f"{np.min(img)}    {np.max(img)}")
    img = img.astype(np.float32) / 255.0
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
    print(f"Img batch: {img_batch.shape}")

    model.build(input_shape=img_batch.shape)
    model_path = 'checkpoints/DexiNed2BIPED/DexiNed23_model.h5'
    model.load_weights(model_path)

    preds = model(img_batch, training=False)
    preds = [tf.sigmoid(i).numpy().squeeze() for i in preds]
    # preds = [i.numpy().squeeze() for i in preds]
    preds = np.array(preds)
    print(f"All preds: {preds.shape}")

    avg = np.mean(preds, axis=0)
    avg[avg < 0.0] = 0.0
    avg = image_normalization(avg).astype(np.uint8)
    avg = (255 - avg)
    # avg = cv2.bitwise_not(np.uint8(image_normalization(avg)))
    print(f"Avg: {avg.shape} {np.min(avg)}  {np.max(avg)}")

    out_filepath = "figures/living_room_edges.png"
    imwrite(out_filepath, avg)

        # Save to final checkpoint name
    filepath = "checkpoints/final_dexined_tf2_model/"
    print(f"Saving exported model to {filepath}")
    # tf.saved_model.save(model, filepath)
