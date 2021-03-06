import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, PReLU, UpSampling2D, MaxPooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization

class Conv(Layer):
    def __init__(self, n_filters, kernel_size=3, non_linearity='relu', spectral_normalization=False, power_iterations=5, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.spectral_normalization = spectral_normalization
        self.power_iterations = power_iterations
        if self.spectral_normalization:
            self.conv = SpectralNormalization(
                Conv2D(
                    filters=self.n_filters,
                    kernel_size=self.kernel_size,
                    padding='same',
                    activation=None,
                ),
                power_iterations=self.power_iterations,
            )
        else:
            self.conv = Conv2D(
                filters=self.n_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=None,
            )
        if self.non_linearity == 'lrelu':
            self.act = LeakyReLU(0.1)
        elif self.non_linearity == 'prelu':
            self.act = PReLU(shared_axes=[1, 2])
        else:
            self.act = Activation(self.non_linearity)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)
        return outputs

class ConvBlock(Layer):
    def __init__(self, n_filters, kernel_size=3, non_linearity='relu', n_non_lins=2, spectral_normalization=False, power_iterations=5, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.n_non_lins = n_non_lins
        self.spectral_normalization = spectral_normalization
        self.power_iterations = power_iterations
        self.convs = [
            Conv(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                spectral_normalization=self.spectral_normalization,
                power_iterations=self.power_iterations,
            ) for _ in range(self.n_non_lins)
        ]

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs

class UpConv(Layer):
    def __init__(self, n_filters, kernel_size=3, spectral_normalization=False, power_iterations=5, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.spectral_normalization = spectral_normalization
        self.power_iterations = power_iterations
        if self.spectral_normalization:
            self.conv = SpectralNormalization(
                Conv2D(
                    filters=self.n_filters,
                    kernel_size=self.kernel_size,
                    padding='same',
                    activation=None,
                ),
                power_iterations=self.power_iterations,
            )
        else:
            self.conv = Conv2D(
                filters=self.n_filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation=None,
            )
        self.up = UpSampling2D(size=(2, 2))

    def call(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv(outputs)
        return outputs


class Unet(Model):
    def __init__(
            self,
            n_output_channels=1,
            kernel_size=3,
            layers_n_channels=[64, 128, 256, 512, 1024],
            layers_n_non_lins=2,
            non_linearity='relu',
            spectral_normalization=False,
            power_iterations=1,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.layers_n_channels = layers_n_channels
        self.n_layers = len(self.layers_n_channels)
        self.spectral_normalization = spectral_normalization
        self.layers_n_non_lins = layers_n_non_lins
        self.non_linearity = non_linearity
        self.power_iterations = power_iterations
        self.down_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
                spectral_normalization=self.spectral_normalization,
                power_iterations=self.power_iterations,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.down = MaxPooling2D(pool_size=(2, 2), padding='same')
        self.bottom_conv = ConvBlock(
            n_filters=self.layers_n_channels[-1],
            kernel_size=self.kernel_size,
            non_linearity=self.non_linearity,
            n_non_lins=self.layers_n_non_lins,
            spectral_normalization=self.spectral_normalization,
            power_iterations=self.power_iterations,
        )
        self.up_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
                spectral_normalization=self.spectral_normalization,
                power_iterations=self.power_iterations,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.ups = [
            UpConv(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                spectral_normalization=self.spectral_normalization,
                power_iterations=self.power_iterations,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        if self.spectral_normalization:            
            self.final_conv = SpectralNormalization(
                Conv2D(
                    filters=self.n_output_channels,
                    kernel_size=1,
                    padding='same',
                    activation=None,
                ),
                power_iterations=self.power_iterations,
            )    
        else:
            self.final_conv = Conv2D(
                filters=self.n_output_channels,
                kernel_size=1,
                padding='same',
                activation=None,
            )
        
    def pad(self, image):
        r"""Convert images to 64x64x1 shaped tensors to feed the model, using zero-padding."""
        pad = tf.constant([[0,0], [6,7],[6,7], [0,0]])
        return tf.pad(image, pad, "CONSTANT")    
        
    def crop(self, image):
        r"""Crop back the image to its original size and convert it to np.array"""
        return tf.image.crop_to_bounding_box(image, 6, 6, 51, 51)

    def call(self, inputs):
        scales = []
        outputs = self.pad(inputs)
        for conv in self.down_convs:
            outputs = conv(outputs)
            scales.append(outputs)
            outputs = self.down(outputs)
        outputs = self.bottom_conv(outputs)
        for scale, conv, up in zip(scales[::-1], self.up_convs[::-1], self.ups[::-1]):
            outputs = up(outputs)
            outputs = tf.concat([outputs, scale], axis=-1)
            outputs = conv(outputs)
        outputs = self.final_conv(outputs)
        outputs = self.crop(outputs)
        return outputs
