import math
import random
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

random.seed(0)

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
    """
    
    """
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
