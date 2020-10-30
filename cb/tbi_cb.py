import numpy as np
import tensorflow as tf
import cv2


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
            ground_truth = np.argmax(label[0], axis=-1)
            ground_truth = (ground_truth * 50).astype(np.uint8)
            ground_truth = self.colorize(ground_truth)
            ground_truth = np.reshape(ground_truth, (-1, input_shape[0], input_shape[1], 3))


            predicted_label = self.model.predict(batch_of_imgs)[0]
            predicted_label = np.argmax(predicted_label, axis=-1)
            predicted_label = (predicted_label * 50).astype(np.uint8)
            pred_img = self.colorize(predicted_label)
            pred_img = np.reshape(pred_img, (-1, input_shape[0], input_shape[1], 3))


            file_writer = tf.summary.create_file_writer(self.path)
            with file_writer.as_default():
                tf.summary.image(self.tag + str(i) + "_x", input_image, step=epoch)
                tf.summary.image(self.tag + str(i) + "_y", ground_truth, step=epoch)
                tf.summary.image(self.tag + str(i) + "_pred", pred_img, step=epoch)

        return

    def colorize(self, img):
        channel2 = np.ones((img.shape[0], img.shape[1], 1)) * 255
        channel3 = np.ones((img.shape[0], img.shape[1], 1)) * 255
        channel1 = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img_hsv = np.concatenate([channel1, channel2, channel3], axis=-1).astype(np.uint8)
        img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_color
