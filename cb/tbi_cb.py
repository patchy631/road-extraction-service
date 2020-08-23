import numpy as np
import tensorflow as tf


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, data_set, tag=""):
        super(TensorBoardImage, self).__init__()
        self.tag = tag
        self.path = output_dir
        self.tb_ds = data_set

    def on_epoch_end(self, epoch, logs={}):
        tb_ds_itr = self.tb_ds.__iter__()

        for i in range(10):
            batch_of_imgs, label = tb_ds_itr.next()
            img = batch_of_imgs * 255
            img = img.numpy()
            input_image = img.astype(np.uint8)

            input_shape = input_image.shape[1:-1]

            label = label.numpy()
            ground_truth = (label[0, :, :, 0] * 255).astype(np.uint8)
            ground_truth = np.reshape(ground_truth, (-1, input_shape[0], input_shape[1], 1))

            predicted_label = self.model.predict(batch_of_imgs)[0]
            predicted_label = np.reshape(predicted_label, (input_shape[0], input_shape[1]))

            final_img = np.zeros((input_shape[0], input_shape[1]), np.uint8)
            thresh_indices = predicted_label[:, :] > 0.5
            final_img[thresh_indices] = 255

            pred_img = np.reshape(final_img, (-1, input_shape[0], input_shape[1], 1))

            file_writer = tf.summary.create_file_writer(self.path)
            with file_writer.as_default():
                tf.summary.image(self.tag + str(i) + "_x", input_image, step=epoch)
                tf.summary.image(self.tag + str(i) + "_y", ground_truth, step=epoch)
                tf.summary.image(self.tag + str(i) + "_pred", pred_img, step=epoch)

        return
