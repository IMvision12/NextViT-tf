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
    #stem_chs, depths, path_dropout
    def __init__(self, stem_chs, depths, path_dropout, num_classes):
        super().__init__()
        
        strides=[1, 2, 2, 2]
        sr_ratios=[8, 4, 2, 1]

        self.stem = tf.keras.Sequential([
            ConvBNReLU(stem_chs[0], kernel_size=3, strides=2),
            ConvBNReLU(stem_chs[1], kernel_size=3, strides=1),
            ConvBNReLU(stem_chs[2], kernel_size=3, strides=1),
            ConvBNReLU(stem_chs[2], kernel_size=3, strides=2),
        ])

        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]
        
        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1] - 1) + [NTB],
                                  [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                                  [NCB] * (depths[3] - 1) + [NTB]]
        
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x for x in tf.linspace(0.0, path_dropout, sum(depths))]
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(output_channel, strides=strides, path_dropout=dpr[idx + block_id],
                                drop=0, head_dim=32)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(output_channel, path_dropout=dpr[idx + block_id], strides=strides,
                                sr_ratio=sr_ratios[stage_id], head_dim=32, mix_block_ratio=0.75,
                                attn_drop=0, drop=0)
                    features.append(layer)
            idx += numrepeat

        self.features = features
        self.norm = layers.BatchNormalization(epsilon=1e-5)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.proj_head = layers.Dense(num_classes, activation='softmax')
        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]

    def call(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.proj_head(x)
        return x
    

def nextvit_small(input_shape=(None, None, 3), num_classes=1000):
    input_layer = layers.Input(input_shape)
    output_layer = NextViT(stem_chs=CONFIG['SMALL']['stem_chs'], 
                           depths=CONFIG['SMALL']['depths'], 
                           path_dropout=CONFIG['SMALL']['drop_path'], 
                           num_classes=1000)(input_layer)
    model = keras.Model(input_layer, output_layer)
    return model

model = nextvit_small((224,224,3))