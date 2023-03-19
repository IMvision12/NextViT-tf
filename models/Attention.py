import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Union
import numpy as np

EPSILON = 1e-5


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class E_MHSA(layers.Layer):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(
        self,
        dim,
        out_dim=None,
        head_dim=32,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0.0,
        sr_ratio=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.q = tf.keras.layers.Dense(dim, use_bias=qkv_bias)
        self.k = tf.keras.layers.Dense(dim, use_bias=qkv_bias)
        self.v = tf.keras.layers.Dense(dim, use_bias=qkv_bias)
        self.proj = tf.keras.layers.Dense(self.out_dim)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = tf.keras.layers.AveragePooling1D(
                pool_size=self.N_ratio, strides=self.N_ratio
            )
            self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-5)

    def call(self, x):
        B = shape_list(x)[0]
        N = shape_list(x)[1]
        C = shape_list(x)[2]
        q = self.q(x)
        q = tf.reshape(q, (B, N, self.num_heads, int(C // self.num_heads)))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = tf.transpose(x, perm=[0, 2, 1])
            x_ = self.sr(x_)
            x_ = tf.transpose(x_, perm=[0, 2, 1])
            k = self.k(x_)
            k = tf.reshape(k, (B, -1, self.num_heads, C // self.num_heads))
            k = tf.transpose(k, perm=[0, 2, 3, 1])
            v = self.v(x_)
            v = tf.reshape(v, (B, -1, self.num_heads, C // self.num_heads))
            v = tf.transpose(v, perm=[0, 2, 1, 3])
        else:
            k = self.k(x)
            k = tf.reshape(k, (B, -1, self.num_heads, C // self.num_heads))
            k = tf.transpose(k, perm=[0, 2, 3, 1])
            v = self.v(x)
            v = tf.reshape(v, (B, -1, self.num_heads, C // self.num_heads))
            v = tf.transpose(v, perm=[0, 2, 1, 3])
        attn = tf.matmul(q, k) * self.scale

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Testing
e_mhsa = E_MHSA(
    dim=64,
    out_dim=128,
    head_dim=32,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.1,
    proj_drop=0.1,
    sr_ratio=1,
)
sample_ip_tf = tf.random.normal(shape=(4, 16, 64))
print(e_mhsa(sample_ip_tf).shape)
