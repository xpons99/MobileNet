import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Concatenate,
    Reshape,
    MultiHeadAttention,
    LayerNormalization,
)
from typing import Tuple


def sin_cos_position_encodings(len, dim, n=10000):
    if dim % 2 != 0:
        raise Exception("dim must be a multiple of 2")
    encodings = np.zeros((len, dim))
    for i in range(len):
        for j in np.arange(int(dim / 2)):
            theta = i / n ** (2 * j / dim)
            encodings[i, 2 * j] = np.sin(theta)
            encodings[i, 2 * j + 1] = np.cos(theta)
    return encodings


class WithSpatialPositionalEncodings(Layer):
    """Adds spatial embeddings to a spatial feature map.

    Takes as input a (batch, height, width, channels) feature map and channel
    concatenates positional encodings for both X and Y spatial information;
    i.e. positional encodings for both [:,i,:,:] and [:,:,i,:] are standard positional
    encodings for an ith element

    Args:
        encoding_dim: dimensionality of generated positional sin/cosine encodings. since two
            sets of encodings are added, one for each of X and Y, the result is an additional
            2x encoding_dim channels.

    Raises:
        Exception: during `build` if inputs are not 4D or height != width.
    """

    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim

    def get_config(self):
        config = super().get_config()
        config.update({"encoding_dim": self.encoding_dim})
        return config

    def build(self, input_shape: Tuple[int]):
        # check input shape
        if len(input_shape) != 4:
            raise Exception(
                "WithSpatialPositionalEncodings only supports a 4D feature map of"
                " shape (batch, width, height, channels)"
            )
        _batch, height, width, _channels = input_shape
        if height != width:
            raise Exception(
                "WithSpatialPositionalEncodings currently only supports square input,"
                f" not {input_shape}"
            )
        self.hw = height

        # create base sin/cosine embeddings
        encodings = sin_cos_position_encodings(self.hw, self.encoding_dim)

        # tile along one axis, then the other, then combine
        tiled_x = np.tile(encodings, reps=(self.hw, 1, 1))
        tiled_y = np.transpose(tiled_x, (1, 0, 2))
        encodings = np.concatenate([tiled_x, tiled_y], axis=2)

        # add leading batch dim in prep for broadcast and convert to TF tensor
        encodings = np.expand_dims(encodings, axis=0)
        self.encodings = tf.convert_to_tensor(encodings)

    def call(self, inputs):
        # broadcast along (now known) inputs batch dimension
        broadcast_shape = tf.where(
            [True, False, False, False], tf.shape(inputs), [-1, self.hw, self.hw, 4]
        )
        encodings = tf.broadcast_to(self.encodings, broadcast_shape)
        # channel concatenate with inputs
        return Concatenate()([inputs, encodings])


class PatchAttention(Layer):
    """Performs a patch based self attention operation.

    Takes as input a (batch, height, width, channels) feature map, reshapes to
    (height*width, channels) ( i.e. an effective patch_size of 1 ), applies layer
    normalization and cross self attention before reshaping back to the original input
    shape.

    note:
    * MultiHeadAttention uses num_heads=1 for broader compatability with resulting graph.

    Args:
        key_dim: key dimension configuration for MultiHeadAttention
        attention_dropout: dropout configuration for MultiHeadAttention

    Raises:
        Exception: during `build` if inputs are not 4D or height != width.
    """

    def __init__(self, key_dim: int, attention_dropout: float = 0.1):
        super().__init__()
        self.key_dim = key_dim
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {"key_dim": self.key_dim, "attention_dropout": self.attention_dropout}
        )
        return config

    def build(self, input_shape: Tuple[int]):
        if len(input_shape) != 4:
            raise Exception(
                "PatchAttention only supports a 4D feature map of shape (batch, width, height, channels)"
            )
        _batch, width, height, channels = input_shape

        # first operation will be to flatten spatial feature map into 1D vector
        # in preperation for self attention. note: this is equivalent to a patch_size
        # of 1.
        patch_size = 1
        num_patches = int((width * height) / patch_size)
        patch_shape = (num_patches, channels)
        self.reshape_to_patches = Reshape(patch_shape, name="to_patches")

        # next operations will (standard) layer norm, cross attention
        # note: we explicitly only support single head attention
        self.layer_norm = LayerNormalization()
        self.cross_attention = MultiHeadAttention(
            num_heads=1, key_dim=self.key_dim, dropout=self.attention_dropout
        )

        # and finally a reshape back to original spatial shape
        self.reshape_from_patches = Reshape(
            (width, height, channels), name="from_patches"
        )

    def call(self, inputs):
        patches = self.reshape_to_patches(inputs)
        patches = self.layer_norm(patches)
        attended_patches = self.cross_attention(patches, patches)
        return self.reshape_from_patches(attended_patches)
