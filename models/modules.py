import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Attention import E_MHSA

EPSILON = 1e-5


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class ConvBNReLU(layers.Layer):
    def __init__(self, filters, kernel_size, strides, groups=1):
        super().__init__()
        self.conv = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            groups=groups,
        )
        self.norm = layers.BatchNormalization(epsilon=EPSILON)
        self.act = layers.Activation("relu")

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MHCA(layers.Layer):
    def __init__(self, filters, head_dim):
        super().__init__()
        self.group_conv3x3 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            groups=filters // head_dim,
        )
        self.norm = layers.BatchNormalization(epsilon=EPSILON)
        self.act = layers.Activation("relu")
        self.projection = layers.Conv2D(filters, kernel_size=1)

    def forward(self, x):
        x = self.group_conv3x3(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.projection(x)
        return x


# https://github.com/bytedance/Next-ViT/blob/main/classification/nextvit.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(layers.Layer):
    def __init__(self, filters, strides=1):
        super().__init__()
        self.avgpool = layers.AveragePooling2D((2, 2), strides=2)
        self.conv = layers.Conv2D(filters, (1, 1))

    def call(self, x):
        return self.conv(self.avgpool(x))


class mlp(layers.Layer):
    def __init__(self, filters, mlp_ratio=None, drop=0.0):
        super().__init__()
        hidden_dim = _make_divisible(filters * mlp_ratio, 32)
        self.conv1 = layers.Conv2D(hidden_dim, kernel_size=1)
        self.act = layers.Activation("relu")
        self.conv2 = layers.Conv2D(hidden_dim, kernel_size=1)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class NCB(layers.Layer):
    def __init__(
        self,
        filters,
        strides=1,
        path_dropout=0,
        drop=0,
        head_dim=32,
        mlp_ratio=3,
    ):
        super().__init__()
        self.filters = filters
        self.patch_embed = PatchEmbed(filters, strides)
        self.mhca = MHCA(filters, head_dim)
        self.attention_path_dropout = StochasticDepth(path_dropout)
        self.mlp = mlp(filters, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = StochasticDepth(path_dropout)

    def call(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class NTB(layers.Layer):
    def __init__(
        self,
        filters,
        path_dropout,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        drop=0,
    ):
        super().__init__()
        self.mix_block_ratio = mix_block_ratio
        self.filters = filters

        self.mhsa_out_channels = _make_divisible(
            int(filters * mix_block_ratio), 32
        )
        self.mhca_out_channels = filters - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(self.mhsa_out_channels, stride)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mhsa_drop_path = StochasticDepth(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhca_out_channels, strides=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_drop_path = StochasticDepth(
            path_dropout * (1 - mix_block_ratio)
        )

        self.mlp = mlp(filters, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_drop_path = StochasticDepth(path_dropout)

    def call(self, x):
        x = self.patch_embed(x)
        out = x
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + out
        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = tf.concat([x, out], axis=1)
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x
