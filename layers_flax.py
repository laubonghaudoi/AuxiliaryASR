"""
"""

import math
import random
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

random.seed(0)
# Layer definitions


class LinearNorm(nnx.Module):
    """Linear layer with normalization."""

    def __init__(self, in_dim, out_dim, bias=True, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()

        initializer = nnx.initializers.xavier_uniform()
        self.linear_layer = nnx.Linear(
            in_dim,
            out_dim,
            use_bias=bias,
            kernel_init=initializer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        return self.linear_layer(x)


class ConvNorm(nnx.Module):
    """Conv layer with normalization."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        strides=1,
        padding=None,
        dilation=1,
        bias=True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        initializer = nnx.initializers.xavier_uniform()

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_dilation=dilation,
            use_bias=bias,
            kernel_init=initializer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        return self.conv(x)


class CausualConv(nnx.Module):
    """Conv layer with normalization."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        strides=1,
        padding=1,
        dilation=1,
        bias=True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2

        initializer = nnx.initializers.xavier_uniform()

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            kernel_dilation=dilation,
            use_bias=bias,
            kernel_init=initializer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        x = self.conv(x)
        x = x[:, :, : -self.padding]
        return x


def _get_activation(activ):
    """Get activation function."""
    if activ == 'lrelu':
        return nnx.leaky_relu
    elif activ == 'relu':
        return nnx.relu
    elif activ == 'gelu':
        return nnx.gelu
    else:
        raise ValueError(f'Activation {activ} not supported')


# Block definitions


class CausualBlock(nnx.Module):
    """Causual block."""

    def _get_conv(
        self,
        hidden_dim,
        dilation,
        activ='lrelu',
        dropout_p=0.2,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        layers = [
            CausualConv(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            _get_activation(activ),
            nnx.BatchNorm(num_features=hidden_dim, rngs=rngs),
            nnx.Dropout(rate=dropout_p),
            CausualConv(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1
            ),
            _get_activation(activ),
            nnx.Dropout(rate=dropout_p),
        ]
        return nnx.Sequential(layers)

    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        super().__init__()

        self.blocks = [
            self._get_conv(
                hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p
            )
            for i in range(n_conv)
        ]

    def __call__(self, x: jax.Array):
        for block in self.blocks:
            res = x
            x = block(x)
            x = x + res
        return x


class ConvBlock(nnx.Module):
    """Conv block."""

    def _get_conv(
        self,
        hidden_dim,
        dilation,
        activ='lrelu',
        dropout_p=0.2,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        layers = [
            ConvNorm(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            _get_activation(activ),
            nnx.GroupNorm(
                num_features=hidden_dim, num_groups=self._n_groups, rngs=rngs
            ),
            nnx.Dropout(rate=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation(activ),
            nnx.Dropout(rate=dropout_p),
        ]
        return nnx.Sequential(layers)

    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        super().__init__()

        self._n_groups = 8
        self.blocks = [
            self._get_conv(
                hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p
            )
            for i in range(n_conv)
        ]

    def __call__(self, x: jax.Array):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x


class LocationLayer(nnx.Module):
    """Location Layer."""

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()

        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=int((attention_kernel_size - 1) / 2),
            strides=1,
            dilation=1,
            bias=False,
        )
        self.location_dense = LinearNorm(
            attention_n_filters,
            attention_dim,
            bias=False,
        )

    def __call__(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nnx.Module):
    """Attention mechanism"""

    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super().__init__()

        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float('inf')

    def get_alignment_energies(
        self, query, processed_memory, attention_weights_cat
    ):
        """PARAMS

        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            jnp.tanh(
                processed_query + processed_memory + processed_attention_weights
            )
        )
        energies = jnp.squeeze(energies, axis=-1)
        return energies

    def __call__(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """PARAMS

        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment = jnp.where(mask, self.score_mask_value, alignment)

        attention_weights = nnx.softmax(alignment, axis=1)
        attention_context = jnp.matmul(
            jnp.expand_dims(attention_weights, axis=1), memory
        )
        attention_context = jnp.squeeze(attention_context, axis=1)

        return attention_context, attention_weights


class ForwardAttentionV2(nnx.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super().__init__()

        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(
            self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            jnp.tanh(processed_query + processed_memory + processed_attention_weights)
        )
        energies = jnp.squeeze(energies, axis=-1)
        return energies

    def __call__(self,
                 attention_hidden_state,
                 memory,
                 processed_memory,
                 attention_weights_cat,
                 mask,
                 log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)
        if mask is not None:
            log_energy = jnp.where(mask, self.score_mask_value, log_energy)

        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        for sft in range(2):
            shifted = log_alpha[:, :max_time - sft]
            shift_padded = jnp.pad(shifted, (sft, 0), 'constant', self.score_mask_value)
            log_alpha_shift_padded.append(
                jnp.expand_dims(shift_padded, axis=2)
            )

        biased = jax.nn.logsumexp(
            jnp.concatenate(log_alpha_shift_padded, 2), 2
        )

        log_alpha_new = biased + log_energy

        attention_weights = nnx.softmax(log_alpha_new, dim=1)

        attention_context = jnp.matmul(
            jnp.expand_dims(attention_weights, axis=1), memory
        )
        attention_context = jnp.squeeze(attention_context, axis=1)

        return attention_context, attention_weights, log_alpha_new


class PhaseShuffle2d(nnx.Module):
    def __init__(self, n=2):
        super().__init__()

        self.n = n
        self.random = random.Random(1)

    def __call__(self, x: jax.Array, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = jnp.concatenate([right, left], dim=3)
        return shuffled


def create_dct_matrix(n_mfcc: int, n_mels: int, norm: Optional[str] = None) -> jnp.ndarray:
    """Creates a DCT transformation matrix (JAX version).

    This function replicates torchaudio.functional.create_dct using JAX,
    handling the normalization correctly.

    Args:
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of mel filterbanks.
        norm: Normalization mode ('ortho' or None).

    Returns:
        A JAX array representing the DCT matrix (n_mfcc, n_mels).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    n = jnp.arange(float(n_mels))
    k = jnp.expand_dims(
        jnp.arange(float(n_mfcc)),
        axis=1
    )
    dct_matrix = jnp.cos(math.pi / float(n_mels) * (n + 0.5) * k)

    if norm is None:
        dct_matrix = dct_matrix * 2.0
    else:
        dct_matrix = dct_matrix.at[0].multiply(1.0 / math.sqrt(2.0))
        dct_matrix = dct_matrix * jnp.sqrt(2.0 / float(n_mels))

    return dct_matrix.T


class MFCC(nnx.Module):
    def __init__(self, n_mfcc=40, n_mels=80):  # Correct way of adding rngs
        super().__init__()

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = create_dct_matrix(self.n_mfcc, self.n_mels, self.norm)
        self.dct_mat = nnx.Variable(dct_mat, collection='params')  # Make it a Variable, you can put it into other collections e.g., 'buffers' if needed.

    def __call__(self, mel_specgram):  # Use __call__ instead of forward
        if len(mel_specgram.shape) == 2:
            mel_specgram = jnp.expand_dims(mel_specgram, axis=0)
            unsqueezed = True
        else:
            unsqueezed = False

        # Shape after unsqueeze: (batch, n_mels, time)
        # Need to transpose to (batch, time, n_mels) for matmul
        # DCT matrix shape should be (n_mels, n_mfcc)
        # Result will be (batch, time, n_mfcc)
        mel_transposed = jnp.transpose(mel_specgram, (0, 2, 1))
        mfcc = jnp.matmul(mel_transposed, self.dct_mat.value)

        # Transpose back to (batch, n_mfcc, time)
        mfcc = jnp.transpose(mfcc, (0, 2, 1))

        if unsqueezed:
            mfcc = jnp.squeeze(mfcc, axis=0)
        return mfcc
