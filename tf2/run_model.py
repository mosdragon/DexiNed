from __future__ import absolute_import, division, print_function

import time, os
import numpy as np
from os.path import join
import cv2 as cv

from model import *
from utls import image_normalization,visualize_result, tensor2image, cv_imshow,h5_writer
from dataset_manager import DataLoader

BUFFER_SIZE = 448

SEED = 10
np.random.seed(SEED)

class run_DexiNed():

    def __init__(self, args):
        self.model_state= args.model_state
        self.args = args
        self.img_width=args.image_width
        self.img_height = args.image_height
        self.epochs = args.max_epochs
        self.bs = args.batch_size

    def train(self):
        # Validation and Train dataset generation

        train_data = DataLoader(data_name=self.args.data4train, arg=self.args)
        n_train =train_data.indices.size #data_cache["n_files"]
        val_data = DataLoader(data_name=self.args.data4train,
                              arg=self.args, is_val=True)
        val_idcs = np.arange(val_data.indices.size)
        # Summary and checkpoint manager
        model_dir =self.args.model_name+'2'+self.args.data4train
        summary_dir = os.path.join('logs',model_dir)
        train_log_dir=os.path.join(summary_dir,'train')
        val_log_dir =os.path.join(summary_dir,'test')

        checkpoint_dir = os.path.join(self.args.checkpoint_dir,model_dir)
        epoch_ckpt_dir = checkpoint_dir + 'epochs'
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        os.makedirs(train_log_dir,exist_ok=True)
        os.makedirs(val_log_dir,exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_writer = tf.summary.create_file_writer(train_log_dir)
        val_writer = tf.summary.create_file_writer(val_log_dir)

        my_model = DexiNed(rgb_mean=self.args.rgbn_mean)#rgb_mean=self.args.rgbn_mean

        accuracy = metrics.SparseCategoricalAccuracy()
        accuracy_val = metrics.SparseCategoricalAccuracy()
        loss_bc = losses.BinaryCrossentropy()
        optimizer = optimizers.Adam(
            learning_rate=self.args.lr, beta_1=self.args.beta1)
        iter = 0

        imgs_res_folder = os.path.join(self.args.output_dir,"current_training")
        os.makedirs(imgs_res_folder,exist_ok=True)
        global_loss = 1000.
        t_loss=[]
        ckpt_save_mode = "h5"
        for epoch in range(self.args.max_epochs):
            # training
            t_loss = []
            for step, (x,y) in enumerate(train_data):

                with tf.GradientTape() as tape:
                    pred = my_model(x, training=True)

                    preds,loss = pre_process_binary_cross_entropy(
                        loss_bc,pred, y,self.args, use_tf_loss=False)

                accuracy.update_state(y_true=y,y_pred=preds[-1])
                gradients = tape.gradient(loss, my_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

                # logging the current accuracy value so far.
                t_loss.append(loss.numpy())
                if step%10==0:
                    print("Epoch:", epoch, "Step:",step,"Loss: %.4f"%loss.numpy(),
                          "Accuracy: %.4f"%accuracy.result(),time.ctime())

                if step % 50 == 0:
                    # visualize preds
                    img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                        .format(epoch, step, n_train//self.args.batch_size, loss.numpy())
                    vis_imgs = visualize_result(
                    x=x[2],y=y[2],p=preds,img_title=img_test)
                    cv.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
                if (step)%100==0 and loss < global_loss:
                    save_ckpt_path=os.path.join(checkpoint_dir, "DexiNedL_model.h5")
                    Model.save_weights(my_model,save_ckpt_path,save_format='h5')

                    global_loss = loss
                    print("Model saved in:  ",save_ckpt_path, "Current loss:",global_loss.numpy())

                iter += 1  # global iteration

            t_loss = np.array(t_loss)
            # train summary
            with train_writer.as_default():
                tf.summary.scalar('loss', t_loss.mean(), step=epoch)
                tf.summary.scalar('accuracy', accuracy.result(), step=epoch)

            Model.save_weights(my_model, os.path.join(epoch_ckpt_dir,"DexiNed{}_model.h5".format(str(epoch))),
                                   save_format=ckpt_save_mode)
            print("Epoch:", epoch, "Model saved in Loss: ", t_loss.mean())

            # validation
            val_idx = np.int(np.random.choice(val_idcs,1))
            x_val, y_val= val_data.__getitem__(val_idx)
            pred_val = my_model(x_val)
            v_logits, V_loss = pre_process_binary_cross_entropy(
                loss_bc, pred_val, y_val, self.args, use_tf_loss=False)
            accuracy_val.update_state(y_true=y_val, y_pred=v_logits[-1])
            val_acc = accuracy_val.result()
            print("Epoch(validation):", epoch, "Val loss: ", V_loss.numpy(),
                  "Accuracy: ", val_acc.numpy())
            # validation summary
            with val_writer.as_default():
                tf.summary.scalar('loss', V_loss.numpy(), step=epoch)
                tf.summary.scalar('accuracy', val_acc, step=epoch)

            # Reset metrics every epoch
            accuracy.reset_states()
            accuracy_val.reset_states()

        my_model.summary()


    def test(self):
        # Test dataset generation

        test_data = DataLoader(data_name=self.args.data4test, arg=self.args)
        n_test = test_data.indices.size  # data_cache["n_files"]

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.lr, beta_1=self.args.beta1)

        my_model = DexiNed(rgb_mean=self.args.rgbn_mean)
        input_shape=test_data.input_shape
        my_model.build(input_shape=input_shape)# rgb_mean=self.args.rgbn_mean

        checkpoit_dir = os.path.join(self.args.checkpoint_dir,
                                     self.args.model_name + "2" + self.args.data4train)

        my_model.load_weights(os.path.join(checkpoit_dir,"DexiNedL_model.h5"))

        result_dir = os.path.join(
            self.args.output_dir,
            self.args.model_name+'-'+self.args.data4train+"2"+self.args.data4test)
        os.makedirs(result_dir,exist_ok=True)
        save_dir = ['fuse','avrg','h5']
        save_dirs=[]
        for tmp_dir in save_dir:
            os.makedirs(os.path.join(result_dir,tmp_dir),exist_ok=True)
            save_dirs.append(os.path.join(result_dir,tmp_dir))

        total_time=[]
        data_names = test_data.imgs_name
        data_shape = test_data.imgs_shape
        k=0
        for step, (x,y) in enumerate(test_data):

            start_time = time.time()
            preds = my_model(x,training=False)
            tmp_time = time.time()-start_time
            total_time.append(tmp_time)

            preds = [tf.sigmoid(i).numpy() for i in preds]
            all_preds = np.array(preds)
            for i in range(all_preds.shape[1]):
                tmp_name = data_names[k]
                tmp_name, _ = os.path.splitext(tmp_name)
                tmp_shape = data_shape[k]

                tmp_preds = all_preds[:,i,...]
                tmp_av = np.expand_dims(tmp_preds.mean(axis=0), axis=0)
                tmp_preds = np.concatenate((tmp_preds, tmp_av), axis=0)
                res_preds = []
                for j in range(tmp_preds.shape[0]):
                    tmp_pred = tmp_preds[j, ...]
                    tmp_pred[tmp_pred < 0.0] = 0.0
                    tmp_pred = cv.bitwise_not(np.uint8(image_normalization(tmp_pred)))
                    h, w = tmp_pred.shape[:2]
                    if h != tmp_shape[0] or w != tmp_shape[1]:
                        tmp_pred = cv.resize(tmp_pred, (tmp_shape[1], tmp_shape[0]))
                    res_preds.append(tmp_pred)
                for idx in range(len(save_dirs) - 1):
                    s_dir = save_dirs[idx]
                    tmp = res_preds[6+idx]
                    cv.imwrite(join(s_dir, tmp_name + '.png'), tmp)
                h5_writer(path=join(save_dirs[-1], tmp_name + '.h5'),
                          vars=np.squeeze(res_preds))
                print("saved:", join(save_dirs[-1], tmp_name + '.h5'), tmp_preds.shape)
                k += 1

            # tmp_name = data_names[step][:-3]+"png"
            # tmp_shape = data_shape[step]
            # tmp_path = os.path.join(result_dir,tmp_name)
            # tensor2image(preds[-1].numpy(), img_path =tmp_path,img_shape=tmp_shape)

        total_time = np.array(total_time)

        print('-------------------------------------------------')
        print("End testing in: ", self.args.data4test)
        print("Batch size: ", self.args.test_bs)
        print("Time average per image: ", total_time.mean(), "secs")
        print("Total time: ", total_time.sum(),"secs")
        print('-------------------------------------------------')
        my_model.summary()