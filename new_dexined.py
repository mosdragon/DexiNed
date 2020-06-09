""" DexiNed architecture description


Created by: Xavier Soria Poma
Modified from: https://github.com/machrisaa/tensorflow-vgg
Autonomous University of Barcelona-Computer Vision Center
xsoria@cvc.uab.es/xavysp@gmail.com
"""

import time
import os

from PIL import Image
import numpy as np
from imageio import imread, imwrite

# %tensorflow 1.15
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2

slim = tf.contrib.slim

class DexinedModel():

    # PRETRAINED_MODEL_PATH = "/home/mxs8x15/code/new_hed/DexiNed/checkpoints/DXN_BIPED/train_2/DXN-149999"
    PRETRAINED_MODEL_PATH = "/home/mxs8x15/code/new_hed/DexiNed/checkpoints/DXN_BIPED/train_1/DXN-149736"
    TARGET_H = TARGET_W = 512
    N_CHANNELS = 3

    def __init__(self, checkpoint_path=PRETRAINED_MODEL_PATH):

        self.img_height = self.TARGET_H
        self.img_width = self.TARGET_W
        self.n_channels = self.N_CHANNELS

        print(f"DEBUG: {self.img_height, self.img_width, self.n_channels}" +
            f"{checkpoint_path}")

        # Assume we're in testing mode
        self.images = tf.placeholder(tf.float32,
            [None, self.img_height, self.img_width, self.n_channels])

        self.edgemaps = tf.placeholder(tf.float32,
            [None, self.img_height, self.img_width, 1])

        # Build the network architecture
        self.define_model()

        if checkpoint_path:
            # Session for TensorFlow version 1.
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = InteractiveSession(config=config)

            # Load this model in.
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint_path)


    def predict(self, img):
        """
        Produce the output edgemap.

        Args:
            :img: - an RGB image as a numpy array
        """

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Remove mean RGB value from image.
        R = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        B = np.mean(img[:, :, 2])
        img -= np.array([R, G, B])
        img = cv2.resize(img, dsize=(self.img_width, self.img_height))

        self.setup_testing()

        edge_maps = self.session.run(self.predictions, feed_dict={self.images: [img]})
        single_edge_map = self.get_single_edgemap(img, edge_maps, threshold=0.0)
        return single_edge_map


    def get_single_edgemap(self, img, edge_maps, threshold=0.0):
        edge_maps = [e[0] for e in edge_maps]
        edgemap_avg = np.mean(np.array(edge_maps), axis=0)
        edge_maps.append(edgemap_avg)

        edgemap_fused = edge_maps[-2]
        edgemap_fused[edgemap_fused < threshold] = 0.0

        edgemap_avg[edgemap_avg < threshold] = 0.0

        # Invert colors
        edgemap_fused = 255.0 * (1.0 - edgemap_fused)
        edgemap_avg = 255.0 * (1.0 - edgemap_avg)

        edgemap_fused = np.tile(edgemap_fused, [1, 1, 3])
        edgemap_avg = np.tile(edgemap_avg, [1, 1, 3])

        edgemap_fused = Image.fromarray(np.uint8(edgemap_fused))
        edgemap_avg = Image.fromarray(np.uint8(edgemap_avg))

        img_size = img.shape[:2]
        edgemap_fused = edgemap_fused.resize(img_size)
        edgemap_avg = edgemap_avg.resize(img_size)

        return edgemap_avg


    def setup_testing(self):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """
        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)


    def define_model(self):
        """ DexiNed architecture
        DexiNed is composed by six blocks, the two first blocks have two convolutional layers
        the rest of the blocks is composed by sub blocks and they have 2, 3, 3, 3 sub blocks
        """
        start_time = time.time()
        use_subpixel = False
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        with tf.variable_scope('Xpt') as sc:

            # ------------------------- Block1 ----------------------------------------
            self.conv1_1 = tf.layers.conv2d(self.images, filters=32,
                kernel_size=[3, 3], strides=(2, 2),
                bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="conv1_1",
                kernel_initializer=weight_init) #  bx200x200x32, b=batch size

            self.conv1_1 = slim.batch_norm(self.conv1_1)
            self.conv1_1 = tf.nn.relu(self.conv1_1)

            self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=64,
                kernel_size=[3,3], strides=(1,1),
                bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="conv1_2",
                kernel_initializer=weight_init)  # bx200x200x64

            self.conv1_2 = slim.batch_norm(self.conv1_2)
            self.conv1_2 = tf.nn.relu(self.conv1_2)

            self.output1 = self.side_layer(self.conv1_2, name='output1',
                filters=1, upscale=int(2 ** 1),
                strides=(1,1), kernel_size=[1,1],
                sub_pixel=use_subpixel,
                kernel_init=weight_init)  # bx400x400x1

            self.rconv1 = tf.layers.conv2d(self.conv1_2,
                filters=128, kernel_size=[1,1], activation=None,
                strides=(2,2), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="rconv1",
                kernel_initializer=weight_init)  # bx100x100x128 --Skip left

            self.rconv1 =slim.batch_norm(self.rconv1) # bx100x100x128 --Skip left

            # ------------------------- Block2 ----------------------------------------
            self.block2_xcp = self.conv1_2
            for k in range(1):
                self.block2_xcp = tf.layers.conv2d(
                    self.block2_xcp, filters=128, kernel_size=[3, 3],
                    strides=(1, 1), padding='same',
                    name='conv_block2_{}'.format(k + 1),
                    kernel_initializer=weight_init) # bx200x200x128

                self.block2_xcp = slim.batch_norm(self.block2_xcp)
                self.block2_xcp = tf.nn.relu(self.block2_xcp)

                self.block2_xcp = tf.layers.conv2d(self.block2_xcp,
                    filters=128, kernel_size=[3, 3], strides=(1, 1),
                    padding='same', name='conv2_block2_{}'.format(k + 1),
                    kernel_initializer=weight_init) # bx200x200x128

                self.block2_xcp= slim.batch_norm(self.block2_xcp)

            self.maxpool2_1=slim.max_pool2d(self.block2_xcp, kernel_size=[3,3],
                stride=2, padding='same', scope='maxpool2_1') # bx100x100x128

            self.add2_1 = tf.add(self.maxpool2_1, self.rconv1)# with skip left
            self.output2 = self.side_layer(self.block2_xcp, filters=1,
                name='output2', upscale=int(2 ** 1),
                strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel,
                kernel_init=weight_init) # bx400x400x1

            self.rconv2= tf.layers.conv2d(
                self.add2_1,filters=256, kernel_size=[1,1], activation=None,
                kernel_initializer=weight_init, strides=(2,2),
                bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="rconv2")  # bx50x50x256 # skip left

            self.rconv2 = slim.batch_norm(self.rconv2)  # skip left

            # ------------------------- Block3 ----------------------------------------
            self.block3_xcp = self.add2_1
            self.addb2_4b3 = tf.layers.conv2d(self.maxpool2_1, filters=256,
                kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1),
                bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="add2conv_4b3") # 100x100x256 # skip right

            self.addb2_4b3 = slim.batch_norm(self.addb2_4b3) # skip right

            for k in range(2):

                self.block3_xcp=tf.nn.relu(self.block3_xcp)
                self.block3_xcp = tf.layers.conv2d(
                    self.block3_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='same', name='con1v_block3_{}'.format(k + 1),
                kernel_initializer=weight_init) # bx100x100x256
                self.block3_xcp = slim.batch_norm(self.block3_xcp)
                self.block3_xcp = tf.nn.relu(self.block3_xcp)

                self.block3_xcp = tf.layers.conv2d(
                    self.block3_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1,1),padding='same',name='conv2_block3_{}'.format(k+1),
                kernel_initializer=weight_init)  # bx100x100x256
                self.block3_xcp = slim.batch_norm(self.block3_xcp)
                self.block3_xcp = tf.add(self.block3_xcp, self.addb2_4b3)/2 #  with  right skip

            self.maxpool3_1 = slim.max_pool2d(self.block3_xcp, kernel_size=[3, 3],stride=2, padding='same',
                                             scope='maxpool3_1')  # bx50x50x256
            self.add3_1 = tf.add(self.maxpool3_1, self.rconv2) # with before skip left
            self.rconv3 = tf.layers.conv2d(
                self.add3_1, filters=512, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="rconv3")  # bx25x25x512  # skip left
            self.rconv3 = slim.batch_norm(self.rconv3) # skip left
            self.output3 = self.side_layer(self.block3_xcp, filters=1,name='output3', upscale=int(2 ** 2),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel,
                                           kernel_init=weight_init)   # bx400x400x1

            # ------------------------- Block4 ----------------------------------------
            self.conv_b2b4 = tf.layers.conv2d(
                self.maxpool2_1, filters=256, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="conv_b2b4")  # bx50x50x256 # skip right
            self.block4_xcp= self.add3_1
            self.addb2b3 = tf.add(self.conv_b2b4, self.maxpool3_1)# skip right
            self.addb3_4b4 = tf.layers.conv2d(
                self.addb2b3, filters=512, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="add3conv_4b4")  # bx50x50x512 # skip right
            self.addb3_4b4 = slim.batch_norm(self.addb3_4b4)# skip right
            for k in range(3):
                self.block4_xcp= tf.nn.relu(self.block4_xcp)
                self.block4_xcp = tf.layers.conv2d(
                    self.block4_xcp, filters=512, kernel_size=[3, 3], strides=(1, 1),
                    padding='same', name='conv1_block4_{}'.format(k + 1), kernel_initializer=weight_init)  # bx50x50x512
                self.block4_xcp = slim.batch_norm(self.block4_xcp)
                self.block4_xcp = tf.nn.relu(self.block4_xcp)

                self.block4_xcp = tf.layers.conv2d(
                    self.block4_xcp, filters=512, kernel_size=[3, 3], strides=(1, 1),
                    padding='same', name='conv2_block4_{}'.format(k+1), kernel_initializer=weight_init) # bx50x50x512
                self.block4_xcp = slim.batch_norm(self.block4_xcp)
                self.block4_xcp = tf.add(self.block4_xcp, self.addb3_4b4)/2 #  with  right skip

            self.maxpool4_1 = slim.max_pool2d(self.block4_xcp, kernel_size=[3, 3], stride=2, padding='same',
                                             scope='maxpool3_1')  # bx25x25x728, b=batch size
            self.add4_1 = tf.add(self.maxpool4_1, self.rconv3) # with skip left
            self.rconv4 = tf.layers.conv2d(
                self.add4_1, filters=512, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="rconv4")  # bx25x25x512  # skip leff
            self.rconv4 = slim.batch_norm(self.rconv4)   # skip left

            self.output4 = self.side_layer(self.block4_xcp, filters=1,name='output4', upscale=int(2 ** 3),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel,
                                           kernel_init=weight_init)  # bx400x400x1

            # ------------------------- Block5 ----------------------------------------
            self.convb3_2ab4 = tf.layers.conv2d(
                self.conv_b2b4, filters=512, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                            padding='SAME', name="conv_b2b5")  # bx25x25x512  # skip right

            self.block5_xcp=self.add4_1
            self.addb2b5 =  tf.add(self.convb3_2ab4,self.maxpool4_1)  # skip right
            self.addb2b5 = tf.layers.conv2d(
                self.addb2b5, filters=512, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="addb2b5")  # bx25x25x512# skip right
            self.addb2b5 = slim.batch_norm(self.addb2b5)# skip right
            for k in range(3):
                self.block5_xcp=tf.nn.relu(self.block5_xcp)
                self.block5_xcp= tf.layers.conv2d(
                    self.block5_xcp, filters=512, kernel_size=[3, 3],
                    strides=(1, 1),padding='SAME', name="conv1_block5{}".format(k+1),
                kernel_initializer=weight_init)  # bx25x25x512
                self.block5_xcp = slim.batch_norm(self.block5_xcp)
                self.block5_xcp=tf.nn.relu(self.block5_xcp)

                self.block5_xcp= tf.layers.conv2d(
                    self.block5_xcp, filters=512, kernel_size=[3, 3],
                    strides=(1, 1),padding='SAME', name="conv2_block5{}".format(k+1),
                kernel_initializer=weight_init)  # bx25x25x728
                self.block5_xcp = slim.batch_norm(self.block5_xcp)
                self.block5_xcp=tf.add(self.block5_xcp,self.addb2b5)/2 # wwith  right skip

            self.add5_1 = tf.add(self.block5_xcp, self.rconv4) # with skip left
            self.output5 = self.side_layer(self.block5_xcp, filters=1,name='output5', kernel_size=[1,1],
                                           upscale=int(2 ** 4), sub_pixel=use_subpixel, strides=(1,1),
                                           kernel_init=weight_init)

            # ------------------------- Block6 ----------------------------------------
            self.block6_xcp = self.add5_1
            self.block6_xcp = tf.layers.conv2d(
                self.block6_xcp, filters=256, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="conv0_b6")  # bx25x25x256

            self.block6_xcp = slim.batch_norm(self.block6_xcp)
            self.addb25_2b6 = tf.layers.conv2d(
                self.block5_xcp, filters=256, kernel_size=[1, 1], activation=None,
                kernel_initializer=weight_init, strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                padding='SAME', name="add2b6")  # bx25x25x256# skip right
            self.addb25_2b6 = slim.batch_norm(self.addb25_2b6)# skip right
            for k in range(3):
                self.block6_xcp = tf.nn.relu(self.block6_xcp)
                self.block6_xcp = tf.layers.conv2d(
                    self.block6_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='SAME', name="conv1_block6{}".format(k + 1),
                kernel_initializer=weight_init)  # bx25x25x256
                self.block6_xcp = slim.batch_norm(self.block6_xcp)
                self.block6_xcp = tf.nn.relu(self.block6_xcp)

                self.block6_xcp = tf.layers.conv2d(
                    self.block6_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='SAME', name="conv2_block6{}".format(k + 1),
                kernel_initializer=weight_init)  # bx25x25x256
                self.block6_xcp = slim.batch_norm(self.block6_xcp)
                self.block6_xcp = tf.add(self.block6_xcp, self.addb25_2b6) / 2 #  with  right skip

            self.output6 = self.side_layer(self.block6_xcp, filters=1, name='output6', kernel_size=[1, 1],
                                           upscale=int(2 ** 4), sub_pixel=use_subpixel, strides=(1, 1),
                                           kernel_init=weight_init)
            # ******************** End blocks *****************************************

            self.side_outputs = [self.output1, self.output2, self.output3,
                                 self.output4, self.output5,self.output6]

            self.fuse = tf.layers.conv2d(tf.concat(self.side_outputs, axis=3),filters=1,
                                        kernel_size=[1,1], name='fuse_1',strides=(1,1),padding='same',
                                        kernel_initializer=tf.constant_initializer(1 / len(self.side_outputs)))
            self.outputs = self.side_outputs + [self.fuse]

        print("Build model finished: {:.4f}s".format(time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, inputs, filters=None,kernel_size=None, depth_multiplier=None,
                   padding='same', activation=None, name=None,
                   kernel_initializer=None, strides=(1,1), separable_conv=False):

        if separable_conv:
            conv = tf.layers.separable_conv2d(
                inputs, filters=filters, kernel_size=kernel_size,
                depth_multiplier=depth_multiplier, padding=padding, name=name)
        else:
            conv= tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                             strides=strides,padding=padding, kernel_initializer=kernel_initializer, name=name)
        return conv

    def side_layer(self, inputs, filters=None,kernel_size=None, strides=(1,1),
                   name=None, upscale=None, sub_pixel=False,kernel_init=None):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
        """
        def upsample_block(inputs, filters=None,kernel_size=None, strides=(1,1),
                   name=None, upscale=None, sub_pixel=False):
            i=1
            scale=2
            sub_net=inputs
            output_filters=16
            if sub_pixel is None:
                # Upsampling by transpose_convolution
                while (scale<=upscale):
                    if scale==upscale:

                        sub_net = self.conv_layer(sub_net, filters=filters, kernel_size=kernel_size,
                                                  strides=strides,kernel_initializer=tf.truncated_normal_initializer(mean=0.0),
                                                  name=name + '_conv_{}'.format(i))  # bx100x100x64
                        biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32),
                                             name=name + '_biases_{}'.format(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)

                        sub_net = tf.layers.conv2d_transpose(
                            sub_net, filters=filters, kernel_size=[(upscale), (upscale)],
                            strides=(2, 2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            name='{}_deconv_{}_{}'.format(name, upscale, i)) # upscale/2
                    else:

                        sub_net = self.conv_layer(sub_net, filters=output_filters,
                                                  kernel_size=kernel_size,kernel_initializer=kernel_init,
                                                  strides=strides, name=name + '_conv_{}'.format(i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
                        biases = tf.Variable(tf.constant(0.0, shape=[output_filters], dtype=tf.float32),
                                             name=name + '_biases_{}'.format(i))

                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        # *
                        sub_net = tf.layers.conv2d_transpose(
                            sub_net, filters=output_filters, kernel_size=[(upscale), (upscale)],
                            strides=(2, 2), padding="SAME", kernel_initializer=kernel_init,
                            name='{}_deconv_{}_{}'.format(name, upscale, i))
                    i += 1
                    scale=2**i

            elif sub_pixel is False:
                # Upsampling by bilinear interpolation
                while (scale <= upscale):

                    if scale == upscale:
                        cur_shape = sub_net.get_shape().as_list()
                        sub_net = self.conv_layer(sub_net, filters=1,
                                                  kernel_size=3, kernel_initializer=kernel_init,
                                                  strides=strides, name=name + '_conv'+str(i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
                        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                                             name=name + '_conv_b'+str(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        if cur_shape[1]== self.img_height and cur_shape[2]==self.img_width:
                            pass
                        else:
                            sub_net = self._upscore_layer(input=sub_net,n_outputs=1,stride=upscale,ksize=upscale,
                                                      name=name+'_bdconv'+str(i))
                    else:
                        cur_shape = sub_net.get_shape().as_list()
                        sub_net = self.conv_layer(sub_net, filters=output_filters,
                                                  kernel_size=3, kernel_initializer=kernel_init,
                                                  strides=strides, name=name + '_conv' + str(
                                i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
                        biases = tf.Variable(tf.constant(0.0, shape=[output_filters], dtype=tf.float32),
                                             name=name + '_conv_b' + str(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        if cur_shape[1] == self.img_height and cur_shape[2] == self.img_width:
                            pass
                        else:
                            sub_net = self._upscore_layer(input=sub_net, n_outputs=output_filters, stride=upscale, ksize=upscale,
                                                          name=name + '_bdconv' + str(i))
                    i += 1
                    scale = 2 ** i

            elif sub_pixel is True:
                # Upsampling by subPixel convolution
                while (scale <= upscale):
                    if scale == upscale:
                        sub_net = self.conv_layer(sub_net, filters=4,
                                                  kernel_size=3, kernel_initializer=kernel_init,
                                                  strides=strides, name=name + '_conv'+str(i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
                        biases = tf.Variable(tf.constant(0.0, shape=[4], dtype=tf.float32),
                                             name=name + '_conv_b'+str(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        _err_log = "SubpixelConv2d: The number of input channels == (scale x scale)" \
                                   " x The number of output channels"
                        r = 2
                        if filters >= 1:
                            if int(sub_net.get_shape()[-1]) != int(r ** 2 * filters):
                                raise Exception(_err_log)
                            sub_net = tf.depth_to_space(sub_net, r)
                        else:
                            raise Exception(' the output channel is not setted')
                    else:

                        sub_net = self.conv_layer(
                            sub_net, filters=32, kernel_size=3, kernel_initializer=kernel_init,
                            strides=strides, name=name + '_conv' + str(i))  # bx100x100x32
                        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                             name=name + '_conv_b' + str(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        _err_log = "SubpixelConv2d: The number of input channels == (scale x scale)" \
                                   " x The number of output channels"
                        r = 2
                        sp_filter =8
                        if sp_filter >= 1:
                            if int(sub_net.get_shape()[-1]) != int(r ** 2 * sp_filter):
                                raise Exception(_err_log)
                            sub_net = tf.nn.depth_to_space(sub_net, r)
                        else:
                            raise Exception(' the output channel is not setted')
                    i += 1
                    scale = 2 ** i
            else:
                raise NotImplementedError
            return sub_net
        classifier = upsample_block(inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                        name=name, upscale=upscale, sub_pixel=sub_pixel)
        return classifier

    def _upscore_layer(self, input, n_outputs, name,
                       ksize=4, stride=2,shape=None):
        strides = [1, stride, stride, 1]
        in_features = input.get_shape().as_list()[3]

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(input)
            ot_shape = input.get_shape().as_list()

            h = ((ot_shape[1] - 1) * stride) + 1
            w = ((ot_shape[2] - 1) * stride) + 1
            # new_shape = [in_shape[0], h, w, n_outputs]
            new_shape = [in_shape[0], self.img_height, self.img_width, n_outputs] #output_shape=[,]
        else:
            new_shape = [shape[0], shape[1], shape[2], n_outputs]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, n_outputs, in_features]
        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input) ** 0.5
        weights = self.get_deconv_filter(f_shape,name=name+'_Wb')
        deconv = tf.nn.conv2d_transpose(input,weights, output_shape,
                                        strides=strides, padding='SAME', name=name)
        # _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape,name=''):
        width = f_shape[0]
        heigh = f_shape[0]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)

        return tf.get_variable(name=name, initializer=init, shape=weights.shape)


if __name__ == "__main__":
    img_uri = "data/living_room.png"
    img = imread(img_uri)
    model = DexinedModel()
    em = model.predict(img)

    imwrite("em.png", em)
    imwrite("orig.png", img)
