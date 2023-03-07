import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CONFIG = {
    "SMALL" : {"stem_chs":[64, 32, 64], "depths":[3, 4, 10, 3], "drop_path":0.1},
    "BASE" : {"stem_chs":[64, 32, 64], "depths":[3, 4, 20, 3], "drop_path":0.1},
    "LARGE" : {"stem_chs":[64, 32, 64], "depths":[3, 4, 30, 3], "drop_path":0.1},
}

