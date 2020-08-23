from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, MaxPooling2D, \
    Concatenate, Input, Add, multiply


class UNetPlusPlus:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def conv_block(self, input_tensor, num_filters):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def attention_block(self, input_tensor, att_tensor, n_filters):
        """
        Implementation as mentioned in paper: https://arxiv.org/pdf/1804.03999.pdf
        :param input_tensor:
        :param att_tensor:
        :param n_filters:
        :return:
        """
        g1 = Conv2D(n_filters, kernel_size=(1, 1))(input_tensor)
        x1 = Conv2D(n_filters, kernel_size=(1, 1))(att_tensor)
        attention_tensor = Add()([g1, x1])
        attention_tensor = Activation('relu')(attention_tensor)
        attention_tensor = Conv2D(1, kernel_size=(1, 1))(attention_tensor)
        attention_tensor = Activation('sigmoid')(attention_tensor)
        attention_tensor = multiply([attention_tensor, att_tensor])
        return attention_tensor

    def build_model(self):
        """
        Generates Unet Plus Plus Model (https://arxiv.org/abs/1807.10165)
        :param img_shape:
        :return:
        """
        inputs = Input(shape=self.input_shape)

        block0_0 = self.conv_block(inputs, num_filters=32)
        block1_0 = self.conv_block(MaxPooling2D(strides=2)(block0_0), num_filters=64)
        block0_1 = self.conv_block(Concatenate()([block0_0, UpSampling2D()(block1_0)]), num_filters=32)

        block2_0 = self.conv_block(MaxPooling2D(strides=2)(block1_0), num_filters=128)
        block1_1 = self.conv_block(Concatenate()([block1_0, UpSampling2D()(block2_0)]), num_filters=64)
        block0_2 = self.conv_block(Concatenate()([block0_0, block0_1, UpSampling2D()(block1_1)]), num_filters=32)

        block3_0 = self.conv_block(MaxPooling2D(strides=2)(block2_0), num_filters=256)
        block2_1 = self.conv_block(Concatenate()([block2_0, UpSampling2D()(block3_0)]), num_filters=128)
        block1_2 = self.conv_block(Concatenate()([block1_0, block1_1, UpSampling2D()(block2_1)]), num_filters=64)
        block0_3 = self.conv_block(Concatenate()([block0_0, block0_1, block0_2, UpSampling2D()(block1_2)]),
                                   num_filters=32)

        block4_0 = self.conv_block(MaxPooling2D(strides=2)(block3_0), num_filters=512)
        block3_1 = self.conv_block(Concatenate()([block3_0, UpSampling2D()(block4_0)]), num_filters=256)
        block2_2 = self.conv_block(Concatenate()([block2_0, block2_1, UpSampling2D()(block3_1)]), num_filters=128)
        block1_3 = self.conv_block(Concatenate()([block1_0, block1_1, block1_2, UpSampling2D()(block2_2)]),
                                   num_filters=64)
        block0_4 = self.conv_block(Concatenate()([block0_0, block0_1, block0_2, block0_3, UpSampling2D()(block1_3)]),
                                   num_filters=32)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(block0_4)
        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model
