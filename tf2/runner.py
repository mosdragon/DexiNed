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

        checkpoint_dir = os.path.join('checkpoints' , model_dir)
        epoch_ckpt_dir = os.path.join(checkpoint_dir + 'epochs')
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        tf_writer = tf.summary.create_file_writer(summary_dir)
        model = DexiNedNetwork(rgb_mean=self.RGBN_MEAN)

        accuracy = metrics.SparseCategoricalAccuracy()
        val_accuracy = metrics.SparseCategoricalAccuracy()
        loss_bc = losses.BinaryCrossentropy()
        optimizer = optimizers.Adam(learning_rate=self.init_lr, beta_1=self.beta1)

        global_loss = 1e10
        ckpt_save_mode = "h5"
        step_count = -1

        for epoch in range(self.epochs):
            print(f"Start Training Epoch [{epoch+1}]/[{self.epochs}] @ {time.ctime()}")
            # Train one epoch.
            train_losses = []
            for step, (x, y) in enumerate(train_dataset):
                step_count += 1

                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss = custom_weighted_cross_entropy(pred, y)

                accuracy.update_state(y_true=y, y_pred=pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_losses.extend(loss.numpy().squeeze().tolist())
                batch_loss = np.mean(loss.numpy())

                # Log the current accuracy value so far.
                if (step + 1) % 10 == 0:
                    print(f"[{epoch + 1}/{self.epochs}]: Step {step + 1} " +
                            f"Loss {batch_loss:.3f} " +
                            f"Accuracy: {str(accuracy.result())} " +
                            f" @ {time.ctime()})")

                    with tf_writer.as_default():
                        tf.summary.scalar('batch_loss', batch_loss, step=step_count)
                        tf.summary.scalar('batch_loss_avg', np.mean(train_losses), step=step_count)

                    train_losses = []

                mean_loss = np.mean(train_losses)
                if (step + 1) % 100 == 0 and mean_loss < global_loss:
                    save_ckpt_path = os.path.join(checkpoint_dir, f"model_{epoch}.h5")

                    KerasModel.save_weights(model, save_ckpt_path, save_format='h5')
                    global_loss = mean_loss
                    print(f"Model saved in: {save_ckpt_path} Current loss: {global_loss.numpy()}")
                    # accuracy.reset_states()

            # Post-epoch training summary
            mean_loss = np.mean(train_losses)
            mean_acc = accuracy.result()

            with tf_writer.as_default():
                tf.summary.scalar('loss', mean_loss, step=epoch)
                tf.summary.scalar('accuracy', mean_acc, step=epoch)

            epoch_checkpoint = os.path.join(epoch_ckpt_dir, "DexiNed_epoch{epoch}.h5")
            KerasModel.save_weights(model, epoch_checkpoint, save_format=ckpt_save_mode)
            print(f"Finished [{epoch+1}/{self.epochs}]. " +
                    f"Loss: {str(mean_loss)} Acc: {str(mean_acc)} " +
                    f"@ {time.ctime()}")

            # TODO: Validation.
            val_losses = []
            for step, (x, y) in enumerate(test_dataset):
                pred = model(x, training=False)
                loss = custom_weighted_cross_entropy(pred, y)

                val_accuracy.update_state(y_true=y, y_pred=pred)
                val_losses.extend(loss.numpy().squeeze().tolist())


            # Summarize
            mean_val_loss = np.mean(val_losses)
            mean_val_acc = val_accuracy.result()
            with tf_writer.as_default():
                tf.summary.scalar('val_loss', mean_val_loss, step=epoch)
                tf.summary.scalar('val_accuracy', mean_val_acc, step=epoch)

            print(f"Validation [{epoch + 1}/{self.epochs}]: " +
                    f"Loss {mean_val_loss} " +
                    f"Accuracy: {mean_val_acc} " +
                    f" @ {time.ctime()})")



            # Reset metrics every epoch
            accuracy.reset_states()
            val_accuracy.reset_states()

