# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
# import inspect

import numpy as np
# from termcolor import colored
import tensorflow as tf

from losses import *
from utls.utls import (get_local_time, print_info, print_warning, print_error)

slim = tf.contrib.slim
class xceptionet():

    def __init__(self, args, run='training'):

        self.args = args
        self.utw= self.args.use_trained_model
        self.img_height =args.image_height
        self.img_width =args.image_width

        base_path = os.path.abspath(os.path.dirname(__file__))
        if self.args.use_trained_model:
            if not os.path.exists(base_path) or len(os.listdir(base_path)) == 0:  # :
                self.data_dict=[]
                print(' *** No model to take for fine tuning ***')
            else:
                weights_file = os.path.join(base_path, self.args.model_weights_path)
                self.data_dict = np.load(weights_file, encoding='latin1').item()
                print_info("Model weights loaded from {}".format(self.args.model_weights_path))

        if self.args.use_nir:

            if args.model_state=='test':
                # self.images = tf.placeholder(tf.float32, [None, None,
                #                                           None, self.args.n_channels + 1])
                self.images = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                          self.args.image_width, self.args.n_channels + 1])
            else:
                self.images = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                          self.args.image_width, self.args.n_channels + 1])

        else:
            if args.model_state=='test':
                # self.images = tf.placeholder(tf.float32, [None, None,
                #                                           None, self.args.n_channels])
                self.images = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                          self.args.image_width, self.args.n_channels])
            else:
                self.images = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                          self.args.image_width, self.args.n_channels])

        self.edgemaps = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                    self.args.image_width, 1])

        self.define_model()

    def define_model(self, is_training=True):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """

        start_time = time.time()
        separable=self.args.use_separable_conv
        use_subpixel=self.args.use_subpixel
        # conv1_1
        with tf.variable_scope('Xpt') as sc:

            # conv1 Block1 -----------------------------------------------
            self.conv1_1 = tf.layers.conv2d(self.images, filters=32, kernel_size=[3, 3],
                                        strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                                        padding='SAME', name="conv1_1") #  bx200x200x32, b=batch size
            self.conv1_1 = slim.batch_norm(self.conv1_1)
            self.conv1_1 = tf.nn.relu(self.conv1_1)
            # conv1_2
            self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=64, kernel_size=[3,3],
                                        strides=(1,1), bias_initializer=tf.constant_initializer(0.0),
                                        padding='SAME', name="conv1_2")  # bx200x200x64, b=batch size
            self.conv1_2 = slim.batch_norm(self.conv1_2)
            self.conv1_2 = tf.nn.relu(self.conv1_2)

            self.output1 = self.side_layer(self.conv1_2,name='output1',filters=1, upscale=int(2 ** 1),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel)  # bx400x400x1
            self.rconv1 = tf.layers.conv2d(self.conv1_2,filters=128, kernel_size=[1,1],
                                        activation=None,
                                        kernel_initializer= None,
                                        strides=(2,2), bias_initializer=tf.constant_initializer(0.0),
                                        padding='SAME', name="rconv1")  # bx100x100x128, b=batch size
            self.rconv1 =slim.batch_norm(self.rconv1) # VARIABLE IN NORMALIZATION ****

            # conv2_1 Block2 ------------------------ check separable-----------------------------
            self.block2_xcp = self.conv1_2
            self.add1_1 = tf.layers.conv2d(self.conv1_2, filters=128, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="add1_1_conv")
            for k in range(1):
                self.block2_xcp = self.conv_layer(self.block2_xcp, filters=128, kernel_size=[3, 3],
                                                  depth_multiplier=1, padding='same',
                                                  name='sconv1_block2_{}'.format(k + 1),
                                                  separable_conv=True) if separable else tf.layers.conv2d(
                    self.block2_xcp, filters=128, kernel_size=[3, 3],
                    strides=(1, 1), padding='same', name='conv_block2_{}'.format(k + 1)) # bx200x200x128

                self.block2_xcp = slim.batch_norm(self.block2_xcp)
                self.block2_xcp = tf.nn.relu(self.block2_xcp)
                #conv2_2
                self.block2_xcp = self.conv_layer(self.block2_xcp, filters=128, kernel_size=[3, 3],
                                                  depth_multiplier=1, padding='same',
                                                  name='sconv2_block2_{}'.format(k + 1),
                                                  separable_conv=True) if separable else tf.layers.conv2d(
                                                self.block2_xcp, filters=128, kernel_size=[3, 3],
                                                strides=(1, 1), padding='same', name='conv2_block2_{}'.format(k + 1))
                # bx200x200x128
                self.block2_xcp= slim.batch_norm(self.block2_xcp)
                # self.block2_xcp = tf.add(self.block2_xcp, self.add1_1)/2 # because there is not
                # iteration this line is not needed
            # self.block2_xcp = tf.nn.relu(self.block2_xcp)  # edited 14/03
            self.maxpool2_1=slim.max_pool2d(self.block2_xcp,kernel_size=[3,3],stride=2, padding='same',
                                        scope='maxpool2_1') # bx100x100x128, b=batch size
            self.add2_1 = tf.add(self.maxpool2_1, self.rconv1)
            self.output2 = self.side_layer(self.block2_xcp,filters=1,name='output2', upscale=int(2 ** 1),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel) # bx400x400x1
            self.rconv2= tf.layers.conv2d(self.add2_1,filters=256, kernel_size=[1,1],
                                        activation=None,
                                        kernel_initializer= None,
                                        strides=(2,2), bias_initializer=tf.constant_initializer(0.0),
                                        padding='SAME', name="rconv2")  # bx50x50x256, b=batch size
            self.rconv2 = slim.batch_norm(self.rconv2)

            # conv1 Block3 -----------------------------------------------------
            self.block3_xcp = self.add2_1
            self.addb2_4b3 = tf.layers.conv2d(self.maxpool2_1,filters=256, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="add2conv_4b3") # 100x100x256
            self.addb2_4b3 = slim.batch_norm(self.addb2_4b3)
            for k in range(2):

                self.block3_xcp=tf.nn.relu(self.block3_xcp)

                self.block3_xcp = self.conv_layer(self.block3_xcp, filters=256, kernel_size=[3, 3],
                                                  depth_multiplier=1, padding='same',
                                                  name='sconv1_block3_{}'.format(k + 1),
                                                  separable_conv=True) if separable else tf.layers.conv2d(
                    self.block3_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='same', name='con1v_block3_{}'.format(k + 1)) # bx100x100x256

                self.block3_xcp = slim.batch_norm(self.block3_xcp)
                self.block3_xcp = tf.nn.relu(self.block3_xcp)
                # conv2 group2
                self.block3_xcp = self.conv_layer(self.block3_xcp, filters=256, kernel_size=[3, 3],
                                               depth_multiplier=1, padding='same', name='sconv2_block3_{}'.format(k+1),
                                               separable_conv=True) if separable else tf.layers.conv2d(self.block3_xcp,
                                                filters=256, kernel_size=[3, 3],
                                                strides=(1,1),padding='same',name='conv2_block3_{}'.format(k+1))  # bx100x100x256, b=batch size

                self.block3_xcp = slim.batch_norm(self.block3_xcp)
                self.block3_xcp = tf.add(self.block3_xcp, self.addb2_4b3)/2
                # self.block3_xcp =self.block3_xcp-self.addb2_4b3

            self.block3_xcp = tf.nn.relu(self.block3_xcp) # edited 14/03
            self.maxpool3_1 = slim.max_pool2d(self.block3_xcp, kernel_size=[3, 3],stride=2, padding='same',
                                             scope='maxpool3_1')  # bx50x50x256, b=batch size
            self.add3_1 = tf.add(self.maxpool3_1, self.rconv2)
            self.rconv3 = tf.layers.conv2d(self.add3_1, filters=512, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="rconv3")  # bx25x25x728, b=batch size
            self.rconv3 = slim.batch_norm(self.rconv3) # VARIABLE ON NORMALIZATION ****


            self.output3 = self.side_layer(self.block3_xcp, filters=1,name='output3', upscale=int(2 ** 2),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel)   # bx400x400x1

            # conv1 Block4 -----------------------------------------------------
            # three tomes
            self.conv_b2b4 = tf.layers.conv2d(self.maxpool2_1, filters=256, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="conv_b2b4")  # bx50x50x256
            # use natch normalization in self.conv_b2b4
            self.block4_xcp= self.add3_1
            self.addb2b3 = tf.add(self.conv_b2b4, self.maxpool3_1)
            self.addb3_4b4 = tf.layers.conv2d(self.addb2b3, filters=512, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="add3conv_4b4")  # bx50x50x728
            self.addb3_4b4 = slim.batch_norm(self.addb3_4b4)
            for k in range(3):
                self.block4_xcp= tf.nn.relu(self.block4_xcp)
                self.block4_xcp = tf.layers.separable_conv2d(self.block4_xcp, filters=512, kernel_size=[3, 3],
                                                          depth_multiplier=1, padding='same',
                                                          name='sconv1_block4_{}'.format(k+1)) \
                    if separable else tf.layers.conv2d(self.block4_xcp,
                    filters=512, kernel_size=[3, 3], strides=(1, 1),
                    padding='same', name='conv1_block4_{}'.format(k + 1))  # bx50x50x728, b=batch siz

                self.block4_xcp = slim.batch_norm(self.block4_xcp)
                self.block4_xcp = tf.nn.relu(self.block4_xcp)
                # conv2 group4
                self.block4_xcp = self.conv_layer(self.block4_xcp, filters=512, kernel_size=[3, 3],
                                depth_multiplier=1, padding='same', name='sconv2_block4_{}'.format(k+1),
                                separable_conv=True) if separable else tf.layers.conv2d(self.block4_xcp,
                                                        filters=512, kernel_size=[3, 3], strides=(1, 1),
                                                        padding='same', name='conv2_block4_{}'.format(k+1)) # bx50x50x728, b=batch size
                self.block4_xcp = slim.batch_norm(self.block4_xcp)
                self.block4_xcp = tf.add(self.block4_xcp, self.addb3_4b4)/2
                # self.block4_xcp = self.block4_xcp - self.addb3_4b4
            # end block 4
            self.maxpool4_1 = slim.max_pool2d(self.block4_xcp, kernel_size=[3, 3], stride=2, padding='same',
                                             scope='maxpool3_1')  # bx25x25x728, b=batch size
            self.add4_1 = tf.add(self.maxpool4_1, self.rconv3)
            self.rconv4 = tf.layers.conv2d(self.add4_1, filters=512, kernel_size=[1, 1],
                                           activation=None,
                                           kernel_initializer=None,
                                           strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                           padding='SAME', name="rconv4")  # bx25x25x728*2, b=batch size
            self.rconv4 = slim.batch_norm(self.rconv4)  # VARIABLE ON NORMALIZATION ****

            self.output4 = self.side_layer(self.block4_xcp, filters=1,name='output4', upscale=int(2 ** 3),
                                           strides=(1,1),kernel_size=[1,1],sub_pixel=use_subpixel)  # bx400x400x1
            #self, inputs, filters=None,kernel_size=None, strides=(1,1),name=None, upscale=None, sub_pixel=False):
            # bellow line have to be commented, because it is not used.
            # Just till middle flow is required for this implementation ***

            # conv1 Block5 -----------------------------------------------------
            # self.conv_b2b4 is the maxPooling conv in block 2 ********* check this ********* with
            # before it was  self.conv_b2b4 now it is self.addb3_4b4
            self.convb3_2ab4 = tf.layers.conv2d(self.conv_b2b4, filters=512, kernel_size=[1, 1],
                                              activation=None,
                                              kernel_initializer=None,
                                              strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
                                              padding='SAME', name="conv_b2b5")  # bx25x25x728 before ssmish trainint
            # self.convb3_2ab4 = tf.layers.conv2d(self.maxpool3_1, filters=512, kernel_size=[1, 1],
            #                                     activation=None,
            #                                     kernel_initializer=None,
            #                                     strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
            #                                     padding='SAME', name="conv_b2b5")  # bx25x25x728
            self.block5_xcp=self.add4_1 # self.rconv4

            # self.block5_xcp= tf.add(self.block5_xcp, self.convb2_4b5)
            # self.addb2b5 =  tf.add(self.convb2_4b5,self.maxpool4_1) # this before first ssmihd training
            self.addb2b5 =  tf.add(self.convb3_2ab4,self.maxpool4_1) # this before first ssmihd training
            self.addb2b5 = tf.layers.conv2d(self.addb2b5, filters=512, kernel_size=[1, 1],
                                               activation=None,
                                               kernel_initializer=None,
                                               strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                               padding='SAME', name="addb2b5")  # bx25x25x512
            self.addb2b5 = slim.batch_norm(self.addb2b5)
            for k in range(3):
                self.block5_xcp=tf.nn.relu(self.block5_xcp)
                self.block5_xcp= self.conv_layer(self.block5_xcp,filters=512, kernel_size=[3,3],
                                        depth_multiplier=1,padding='same', name='sconv1_block5{}'.format(k+1),
                                          separable_conv=True) if separable else tf.layers.conv2d(self.block5_xcp, filters=512, kernel_size=[3, 3],
                                            strides=(1, 1),padding='SAME', name="conv1_block5{}".format(k+1))  # bx25x25x728
                self.block5_xcp = slim.batch_norm(self.block5_xcp)
                self.block5_xcp=tf.nn.relu(self.block5_xcp)

                self.block5_xcp= self.conv_layer(self.block5_xcp,filters=512, kernel_size=[3,3],
                                        depth_multiplier=1,padding='same', name='sconv2_block5{}'.format(k+1),
                                          separable_conv=True) if separable else tf.layers.conv2d(self.block5_xcp, filters=512, kernel_size=[3, 3],
                                            strides=(1, 1),padding='SAME', name="conv2_block5{}".format(k+1))  # bx25x25x728
                self.block5_xcp = slim.batch_norm(self.block5_xcp)
                # tmp_add=tf.add(self.add4_1,self.rconv4)
                self.block5_xcp=tf.add(self.block5_xcp,self.addb2b5)/2
                # self.block5_xcp = self.block5_xcp-self.addb2b5

            self.add5_1 = tf.add(self.block5_xcp, self.rconv4)

            self.output5 = self.side_layer(self.block5_xcp, filters=1,name='output5', kernel_size=[1,1],
                                           upscale=int(2 ** 4), sub_pixel=use_subpixel, strides=(1,1))

            # *********************** block 6 *****************************************

            # self.convb2_4b5 = tf.layers.conv2d(self.conv_b2b4, filters=512, kernel_size=[1, 1],
            #                                    activation=None,
            #                                    kernel_initializer=None,
            #                                    strides=(2, 2), bias_initializer=tf.constant_initializer(0.0),
            #                                    padding='SAME', name="conv_b2b5")  # bx25x25x728
            self.block6_xcp = self.add5_1  # self.rconv4
            self.block6_xcp = tf.layers.conv2d(self.block6_xcp, filters=256, kernel_size=[1, 1],
                                               activation=None,
                                               kernel_initializer=None,
                                               strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                               padding='SAME', name="conv0_b6")  # bx25x25x512
            # here maybe batch normalization of block6_xcp
            self.block6_xcp = slim.batch_norm(self.block6_xcp)
            # self.block5_xcp= tf.add(self.block5_xcp, self.convb2_4b5)
            # self.addb45_2b6 = tf.add(self.maxpool4_1, self.block5_xcp) # be careful with this line and the next
            # self.addb25_2b6 = tf.add(self.convb2_4b5, self.block5_xcp) # be careful with this line too
            self.addb25_2b6 = tf.layers.conv2d(self.block5_xcp, filters=256, kernel_size=[1, 1],
                                               activation=None,
                                               kernel_initializer=None,
                                               strides=(1, 1), bias_initializer=tf.constant_initializer(0.0),
                                               padding='SAME', name="add2b6")  # bx25x25x512
            self.addb25_2b6 = slim.batch_norm(self.addb25_2b6)
            for k in range(3):
                self.block6_xcp = tf.nn.relu(self.block6_xcp)
                self.block6_xcp = self.conv_layer(self.block6_xcp, filters=256, kernel_size=[3, 3],
                                                  depth_multiplier=1, padding='same',
                                                  name='sconv1_block6{}'.format(k + 1),
                                                  separable_conv=True) if separable else tf.layers.conv2d(
                    self.block6_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='SAME', name="conv1_block6{}".format(k + 1))  # bx25x25x728
                self.block6_xcp = slim.batch_norm(self.block6_xcp)
                self.block6_xcp = tf.nn.relu(self.block6_xcp)

                self.block6_xcp = self.conv_layer(self.block6_xcp, filters=256, kernel_size=[3, 3],
                                                  depth_multiplier=1, padding='same',
                                                  name='sconv2_block6{}'.format(k + 1),
                                                  separable_conv=True) if separable else tf.layers.conv2d(
                    self.block6_xcp, filters=256, kernel_size=[3, 3],
                    strides=(1, 1), padding='SAME', name="conv2_block6{}".format(k + 1))  # bx25x25x728
                self.block6_xcp = slim.batch_norm(self.block6_xcp)
                # tmp_add=tf.add(self.add4_1,self.rconv4)
                self.block6_xcp = tf.add(self.block6_xcp, self.addb25_2b6) / 2

            self.output6 = self.side_layer(self.block6_xcp, filters=1, name='output6', kernel_size=[1, 1],
                                           upscale=int(2 ** 4), sub_pixel=use_subpixel, strides=(1, 1))

            # ********************            *****************************************
            # ******************** end blocks *****************************************

            self.side_outputs = [self.output1, self.output2, self.output3,
                                 self.output4, self.output5,self.output6]
            # w_shape = [1, 1, len(self.side_outputs), 1]

            self.fuse = tf.layers.conv2d(tf.concat(self.side_outputs, axis=3),filters=1,
                                        kernel_size=[1,1], name='fuse_1',strides=(1,1),padding='same',
                                        kernel_initializer=tf.constant_initializer(1 / len(self.side_outputs)))
            self.outputs = self.side_outputs + [self.fuse]

        print_info("Build model finished: {:.4f}s".format(time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, inputs, filters=None,kernel_size=None, depth_multiplier=None,
                   padding='same', activation=None, name=None,
                   kernel_initializer=None, strides=(1,1), separable_conv=False):

        if separable_conv:
            conv = tf.layers.separable_conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                                      depth_multiplier=depth_multiplier, padding=padding,
                                                      name=name)
        else:
            conv= tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                             strides=strides,padding=padding, kernel_initializer=kernel_initializer, name=name)
        return conv

    def side_layer(self, inputs, filters=None,kernel_size=None, strides=(1,1),
                   name=None, upscale=None, sub_pixel=False):
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
            if not sub_pixel: # for upsamplin by deconv
                while (scale<=upscale):
                    if scale==upscale: # for transpose_convolution
                        sub_net = self.conv_layer(sub_net, filters=filters, kernel_size=kernel_size,
                                                  strides=strides,kernel_initializer=tf.truncated_normal_initializer(mean=0.0),
                                                  name=name + '_conv_{}'.format(i))  # bx100x100x64
                        biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32),
                                             name=name + '_biases_{}'.format(i))
                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        # kernel_size for deconv [3,3]
                        #*
                        sub_net = tf.layers.conv2d_transpose(sub_net, filters=filters,
                                                            kernel_size=[(upscale), (upscale)],
                                                            strides=(2, 2), padding="SAME",
                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                stddev=0.1),
                                                            name='{}_deconv_{}_{}'.format(name, upscale, i)) # upscale/2
                    else:

                        sub_net = self.conv_layer(sub_net, filters=output_filters,
                                                  kernel_size=kernel_size,kernel_initializer=None,
                                                  strides=strides, name=name + '_conv_{}'.format(i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
                        biases = tf.Variable(tf.constant(0.0, shape=[output_filters], dtype=tf.float32),
                                             name=name + '_biases_{}'.format(i))

                        sub_net = tf.nn.bias_add(sub_net, biases)
                        sub_net = tf.nn.relu(sub_net)
                        # *
                        sub_net = tf.layers.conv2d_transpose(sub_net, filters=output_filters,
                                                             kernel_size=[(upscale), (upscale)],
                                                             strides=(2, 2), padding="SAME",
                                                             kernel_initializer=None,
                                                             name='{}_deconv_{}_{}'.format(name, upscale, i)) # tf.truncated_normal_initializer(stddev=0.1)
                    i += 1
                    scale=2**i

            elif sub_pixel is None: # for upsampling by bilinear
                while (scale <= upscale):
                    # Now for bilinear before: for sub pixel upsampling
                    if scale == upscale:
                        cur_shape = sub_net.get_shape().as_list()
                        sub_net = self.conv_layer(sub_net, filters=1,
                                                  kernel_size=3, kernel_initializer=None,
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
                                                  kernel_size=3, kernel_initializer=None,
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

            elif sub_pixel: # for subpixel
                while (scale <= upscale):
                    # Now for bilinear before: for sub pixel upsampling
                    if scale == upscale:
                        sub_net = self.conv_layer(sub_net, filters=4,
                                                  kernel_size=3, kernel_initializer=None,
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

                        sub_net = self.conv_layer(sub_net, filters=32,
                                                  kernel_size=3, kernel_initializer=None,
                                                  strides=strides, name=name + '_conv' + str(
                                i))  # bx100x100x64 tf.truncated_normal_initializer(mean=0.0, stddev=0.15)
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
        # with tf.variable_scope(name):
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

    def setup_testing(self, session):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """
        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """
        self.predictions = []
        self.loss = 0
        self.fuse_output = []
        self.losses=[]

        print_warning('Deep supervision application set to {}'.format(self.args.deep_supervision))
        ci=np.arange(len(self.side_outputs))
        for idx, b in enumerate(self.side_outputs):
            # self.losses.append(cost)
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            if self.args.deep_supervision and idx in ci:
                cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))
                self.loss += (self.args.loss_weights * cost)
                self.predictions.append(output)
            else:
                cost =truth_difference_error(prediction=self.side_outputs[-1], label=self.edgemaps)
                self.loss += (self.args.loss_weights * cost)
                self.predictions.append(output)

        # loss for the last side
        self.fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')  # self by me
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(self.fuse_output)
        # self.losses.append(fuse_cost)
        self.loss += (self.args.loss_weights * fuse_cost)

        pred = tf.cast(tf.greater(self.fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('Training', self.loss)
        tf.summary.scalar('Validation', self.error)

        self.merged_summary = tf.summary.merge_all()
        if self.args.use_nir:
            self.train_log_dir = os.path.join(self.args.logs_dir,
                                          os.path.join(self.args.model_name+'_'+self.args.dataset_name+'_RGBN','train'))
            self.val_log_dir = os.path.join(self.args.logs_dir,
                                        os.path.join(self.args.model_name+'_'+self.args.dataset_name+'_RGBN', 'val'))
        else:
            self.train_log_dir = os.path.join(self.args.logs_dir,
                                          os.path.join(self.args.model_name+'_'+self.args.dataset_name,'train'))
            self.val_log_dir = os.path.join(self.args.logs_dir,
                                        os.path.join(self.args.model_name+'_'+self.args.dataset_name, 'val'))

        if not os.path.exists(self.train_log_dir):
            os.makedirs(self.train_log_dir)
        if not os.path.exists(self.val_log_dir):
            os.makedirs(self.val_log_dir)

        self.train_writer = tf.summary.FileWriter(self.train_log_dir, session.graph)
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)