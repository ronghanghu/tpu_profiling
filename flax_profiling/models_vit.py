# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
clstoken_init = fixed_gaussian_init
posemb_init = fixed_gaussian_init
patch_kernel_init = fixed_gaussian_init
patch_bias_init = fixed_gaussian_init
msa_kernel_init = fixed_gaussian_init
mlp_kernel_init = fixed_gaussian_init
mlp_bias_init = nn.initializers.zeros
head_kernel_init = fixed_gaussian_init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape, self.dtype)
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(
            inputs
        )
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim, dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    droppath_rate: float
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)

        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            kernel_init=msa_kernel_init,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # droppath
        x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_msa")(
            x, deterministic=deterministic
        )
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
        # droppath
        y = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_mlp")(
            y, deterministic=deterministic
        )

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    droppath_rate: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train, encoder_norm=True):
        """Applies Transformer model on the inputs.

        Args:
          inputs: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert inputs.ndim == 3  # (batch, len, emb)

        x = inputs
        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                name="encoderblock_{:02d}".format(lyr),
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1),
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm", dtype=self.dtype)(x) if encoder_norm else x

        return encoded


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    num_classes: int
    patch_size: int
    hidden_size: int
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    droppath_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs

        n, h, w, c = x.shape
        # We can merge s2d+emb into a single conv; it's the same.
        x = nn.Conv(
            name="embedding",
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            kernel_init=patch_kernel_init,
            bias_init=patch_bias_init,
            dtype=self.dtype,
        )(x)

        # Here, x is a grid of embeddings.

        # Transformer.
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # If we want to add a class token, add it here.
        cls = self.param("cls", clstoken_init, (1, 1, c), self.dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

        # we add posemb here
        x = AddPositionEmbs(name="posembed_encoder", posemb_init=posemb_init, dtype=self.dtype)(x)

        x = Encoder(
            name="Transformer",
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            droppath_rate=self.droppath_rate,
            dtype=self.dtype,
        )(x, train=train, encoder_norm=True)

        if self.num_classes:
            x = x[:, 0]
            x = nn.Dense(name="head", features=self.num_classes, dtype=self.dtype, kernel_init=head_kernel_init)(x)
        return x


ViT_B16_nodrop = partial(
    VisionTransformer,
    patch_size=16,
    hidden_size=768,
    num_layers=12,
    mlp_dim=768 * 4,
    num_heads=12,
    dropout_rate=0.0,
    attention_dropout_rate=0.0,
    droppath_rate=0.0,
)
ViT_L16_nodrop = partial(
    VisionTransformer,
    patch_size=16,
    hidden_size=1024,
    num_layers=24,
    mlp_dim=1024 * 4,
    num_heads=16,
    dropout_rate=0.0,
    attention_dropout_rate=0.0,
    droppath_rate=0.0,
)
ViT_H14_nodrop = partial(
    VisionTransformer,
    patch_size=14,
    hidden_size=1280,
    num_layers=32,
    mlp_dim=1280 * 4,
    num_heads=16,
    dropout_rate=0.0,
    attention_dropout_rate=0.0,
    droppath_rate=0.0,
)
