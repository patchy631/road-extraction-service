from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np


class SnapshotCallbackBuilder:
    def __init__(self, sgdr_lr, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.sgdr_lr = sgdr_lr

    def get_callbacks(self):
        callback_list = [
            ModelCheckpoint("./model.hdf5", monitor='val_dice_loss', save_best_only=True, save_weights_only=True,
                            verbose=1),
            self.sgdr_lr
            # LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
