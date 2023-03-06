import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EPSILON =  1e-5

class ConvBNReLU(layers.Layer):
    def __init__(self, filters, kernel_size, strides, groups=1):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', groups=groups)
        self.norm = layers.BatchNormalization(epsilon=EPSILON)
        self.act = layers.Activation('relu')

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

#https://github.com/bytedance/Next-ViT/blob/main/classification/nextvit.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MHCA(layers.Layer):
    def __init__(self, filters, head_dim):
        super().__init__()
        self.group_conv3x3 = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', groups=filters // head_dim)
        self.norm = layers.BatchNormalization(epsilon=EPSILON)
        self.act = layers.Activation('relu')
        self.projection = layers.Conv2D(filters, kernel_size=1)

    def forward(self, x):
        x = self.group_conv3x3(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.projection(x)
        return x
