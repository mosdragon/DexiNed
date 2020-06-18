from __future__ import absolute_import, division, print_function

import time, os
import numpy as np
from os.path import join
import cv2 as cv
from imageio import imread, imwrite
import skimage
import skimage.transform
import skimage.color

from tensorflow.keras.models import Model as KerasModel

from model import *
from utils import image_normalization
# from utils import image_normalization, visualize_result, h5_writer

from data import get_loaders

BUFFER_SIZE = 1024

SEED = 10
np.random.seed(SEED)

class run_DexiNed():
    RGBN_MEAN = [103.939,116.779,123.68, 137.86]

    def __init__(self, epochs=2):
        self.epochs = epochs
        self.init_lr = 0.0002
        self.beta1 = 0.5


    def train(self):
        train_dataset, val_dataset, test_dataset = get_loaders()
        print(f"Loaded datasets")

        # Summary and checkpoint manager
        model_dir = "Dexined_tf2"
        summary_dir = os.path.join('logs', model_dir)
        train_log_dir = os.path.join(summary_dir, 'train')
        val_log_dir = os.path.join(summary_dir, 'test')

        checkpoint_dir = os.path.join('checkpoints' , model_dir)
        epoch_ckpt_dir = os.path.join(checkpoint_dir + 'epochs')
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        os.makedirs(train_log_dir,exist_ok=True)
        os.makedirs(val_log_dir,exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        tf_writer = tf.summary.create_file_writer(summary_dir)
        model = DexiNedNetwork(rgb_mean=self.RGBN_MEAN)

        accuracy = metrics.SparseCategoricalAccuracy()
        accuracy_val = metrics.SparseCategoricalAccuracy()
        loss_bc = losses.BinaryCrossentropy()
        optimizer = optimizers.Adam(learning_rate=self.init_lr, beta_1=self.beta1)

        global_loss = 1e10
        ckpt_save_mode = "h5"
        step_count = -1

        for epoch in range(self.epochs):
            print(f"Beginning Epoch [{epoch+1}]/[{self.epochs}] @ {time.ctime()}")
            # Train one epoch.
            train_losses = []
            for step, (x, y) in enumerate(train_dataset):
                step_count += 1

                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    preds, loss = pre_process_binary_cross_entropy(loss_bc,
                            pred, y, use_tf_loss=False)

                accuracy.update_state(y_true=y, y_pred=preds[-1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_losses.append(loss.numpy())

                # Log the current accuracy value so far.
                if (step + 1) % 10 == 0:
                    print(f"[{epoch + 1}/{self.epochs}]: Step {step + 1} " +
                            f"Loss {loss.numpy():.3f} " +
                            f"Accuracy: {accuracy.result():.3f} " +
                            f" @ {time.ctime()})")

                # if (step + 1) % 30 == 0:
                    with tf_writer.as_default():
                        tf.summary.scalar('batch_loss', loss.numpy(), step=step_count)

                if (step + 1) % 100 == 0 and loss < global_loss:
                    save_ckpt_path = os.path.join(checkpoint_dir, f"DexiNed_model_{epoch}.h5")

                    KerasModel.save_weights(model, save_ckpt_path, save_format='h5')
                    global_loss = loss
                    print(f"Model saved in: {save_ckpt_path} Current loss: {global_loss.numpy()}")

            # Post-epoch training summary
            mean_loss = np.mean(train_losses)
            mean_acc = accuracy.result()

            with tf_writer.as_default():
                tf.summary.scalar('loss', mean_loss, step=epoch)
                tf.summary.scalar('accuracy', mean_acc, step=epoch)

            epoch_checkpoint = os.path.join(epoch_ckpt_dir, "DexiNed_epoch{epoch}.h5")
            KerasModel.save_weights(model, epoch_checkpoint, save_format=ckpt_save_mode)
            print(f"{epoch+1}/{self.epochs} Complete. Loss: {mean_loss} Acc: {mean_acc}")

            # TODO: Validation.

            # Reset metrics every epoch
            accuracy.reset_states()

        # print(model.summary())

