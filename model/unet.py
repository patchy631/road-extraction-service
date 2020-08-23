from tensorflow.keras import layers
from tensorflow.keras import models


class UNet:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)

        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)

        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)

        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)

        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)

        center = self.conv_block(encoder4_pool, 1024)

        decoder4 = self.decoder_block(center, encoder4, 512)

        decoder3 = self.decoder_block(decoder4, encoder3, 256)

        decoder2 = self.decoder_block(decoder3, encoder2, 128)

        decoder1 = self.decoder_block(decoder2, encoder1, 64)

        decoder0 = self.decoder_block(decoder1, encoder0, 32)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        return model
