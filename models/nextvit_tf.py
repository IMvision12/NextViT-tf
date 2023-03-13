import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from modules import NCB, NTB, ConvBNReLU

CONFIG = {
    "SMALL": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 10, 3],
        "drop_path": 0.1,
    },
    "BASE": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 20, 3],
        "drop_path": 0.1,
    },
    "LARGE": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 30, 3],
        "drop_path": 0.1,
    },
}


class NextViT(layers.Layer):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False):
        super().__init__()
        
        self.stem = tf.keras.Sequential([
            ConvBNReLU(stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], kernel_size=3, stride=2),
        ])