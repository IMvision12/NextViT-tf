import tensorflow as tf


class E_MHSA(tf.keras.layers.Layer):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super(E_MHSA, self).__init__()
        self.dim = dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.q = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias)
        self.k = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias)
        self.v = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias)
        self.proj = tf.keras.layers.Dense(self.dim)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = tf.keras.layers.AveragePooling1D(
                pool_size=self.N_ratio, strides=self.N_ratio
            )
            self.norm = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = tf.reshape(q, [B, N, self.num_heads, int(C // self.num_heads)])
        q = tf.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = tf.transpose(x, [0, 2, 1])
            x_ = self.sr(x_)
            if not tf.executing_eagerly() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = tf.transpose(x_, [0, 2, 1])
            k = self.k(x_)
            k = tf.reshape(k, [B, -1, self.num_heads, int(C // self.num_heads)])
            k = tf.transpose(k, [0, 2, 3, 1])
            v = self.v(x_)
            v = tf.reshape(v, [B, -1, self.num_heads, int(C // self.num_heads)])
            v = tf.transpose(v, [0, 2, 1, 3])
        else:
            k = self.k(x)
            k = tf.reshape(k, [B, -1, self.num_heads, int(C // self.num_heads)])
            k = tf.transpose(k, [0, 2, 3, 1])
            v = self.v(x)
            v = tf.reshape(v, [B, -1, self.num_heads, int(C // self.num_heads)])
            v = tf.transpose(v, [0, 2, 1, 3])
        attn = tf.matmul(q, k) * self.scale

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[B, N, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Testing
new_multi = E_MHSA(dim=32)
sample_ip_tf = tf.random.normal(shape=(2, 2, 32))
print(sample_ip_tf.shape)
print(new_multi(sample_ip_tf).shape)