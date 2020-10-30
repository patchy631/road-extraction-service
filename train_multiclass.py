import glob

import numpy as np
import functools
import random
import tensorflow as tf
# import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import tensorflow.keras.backend as K
import segmentation_models as sm
import os
from cb.tbi_cb import TensorBoardImage
from cb.snapshot_cb_builder import SnapshotCallbackBuilder
from cb.sgdr_lr_scheduler import SGDRScheduler


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

img_dir = "aerial-lanes-data/images"
label_dir = "aerial-lanes-data/masks"
train_files = glob.glob(img_dir + "/*.png")
#train_files = train_files[:2000]
train_label_files = []
for x in train_files:
    train_label_files.append(x.replace("images", "masks"))

print(train_label_files[0])
print(train_files[0])

x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(train_files,
                                                                                          train_label_files,
                                                                                          test_size=0.1,
                                                                                          random_state=42)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
img_shape = (2048, 2048, 3)
batch_size = 4
n_classes = 5
epochs = 50
BACKBONE = 'efficientnetb4'

preprocess_input = sm.get_preprocessing(BACKBONE)


def _process_pathnames(fname, label_path):
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_png(img_str, channels=3)

    label_img_str = tf.io.read_file(label_path)
    label_img = tf.image.decode_png(label_img_str)

    label_img = label_img[:, :, 0]
    depth = 5  # depth = number of class including background
    label_img = tf.one_hot(label_img, depth,
           on_value=255.0, off_value=0.0,
           axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random.uniform([],
                                                  -width_shift_range * img_shape[1],
                                                  width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random.uniform([],
                                                   -height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
        # Translate both
        # output_img = tfa.image.translate(output_img,
        #                                  [width_shift_range, height_shift_range])
        # label_img = tfa.image.translate(label_img,
        #                                 [width_shift_range, height_shift_range])
    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random.uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.math.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize(label_img, resize)
        img = tf.image.resize(img, resize)

    brightness_prob = tf.random.uniform([], 0.0, 1.0)
    if tf.math.less(brightness_prob, 0.5):
        img = tf.image.adjust_brightness(img, 0.2)

    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    # img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    label_img = tf.cast(label_img, dtype=tf.float32) * scale
    img = tf.cast(img, dtype=tf.float32) * scale
    return img, label_img


def _tb_augment(img, label_img):
    label_img = tf.image.resize(label_img, [img_shape[0], img_shape[1]])
    img = tf.image.resize(img, [img_shape[0], img_shape[1]])

    label_img = tf.cast(label_img, dtype=tf.float32) * (1 / 255.)
    img = tf.cast(img, dtype=tf.float32) * (1 / 255.)

    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=6,
                         batch_size=batch_size,
                         shuffle=False):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    # print(dataset)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.repeat().batch(batch_size)
    return dataset


def get_tb_dataset(fnames,
                   lnames,
                   preproc_fn=functools.partial(_tb_augment),
                   threads=6,
                   batch_size=1,
                   shuffle=True):
    filenames, labels = zip(*random.sample(list(zip(fnames, lnames)), 300))
    filenames = list(filenames)
    labels = list(labels)
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    # print(dataset)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # dataset = dataset.repeat().batch(batch_size)

    return dataset


tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)
val_ds = get_baseline_dataset(x_val_filenames,
                              y_val_filenames,
                              preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)
tb_ds = get_tb_dataset(x_val_filenames, y_val_filenames)

# LOSSES

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# https://www.programmersought.com/article/60001511310/
def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
         Focal loss for binary classification problems

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


steps_per_epoch = int(np.ceil(num_train_examples / float(batch_size)))
validation_steps = int(np.ceil(num_val_examples / float(batch_size)))

# Sets up a timestamped log directory.
logdir = "logs/train_data/"

# Callbacks
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
tbi_callback = TensorBoardImage(logdir, data_set=tb_ds)

sgdr_lr = SGDRScheduler(min_lr=1e-5,
                        max_lr=1e-3,
                        steps_per_epoch=steps_per_epoch,
                        swa_path='swa_weights/model_swa_{}.hdf5',
                        tb_log_dir=logdir,
                        lr_decay=0.9,
                        cycle_length=5,
                        mult_factor=1.5)

snapshot = SnapshotCallbackBuilder(sgdr_lr=sgdr_lr, nb_epochs=epochs, nb_snapshots=1, init_lr=1e-3)
snapshot_callbacks = snapshot.get_callbacks()
#earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
# DEFINE MODEL
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = sm.Unet(BACKBONE, classes=n_classes, activation='softmax')
  

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0, 1, 1, 1, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
#focal_loss = sm.losses.BinaryFocalLoss()
#dice_loss = dice_coef_loss()
#focal_loss = binary_focal_loss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(threshold=0.5),
          sm.metrics.Recall(threshold=0.5), dice_loss]

model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

model.summary()
#model.load_weights('swa_weights_evaluated/model_swa_4.hdf5')
history = model.fit(train_ds, validation_data=val_ds, validation_steps=validation_steps, epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[tbCallBack, tbi_callback] + snapshot_callbacks
                    )
final_model_path = "model_train_end.hdf5"
model.save(final_model_path)
print(history)
