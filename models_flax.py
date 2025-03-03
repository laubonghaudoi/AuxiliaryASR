import math
import random
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from layers import LinearNorm
from layers_flax import Attention

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

class ASRS2S(nnx.Module):
    """
    The ASRS2S module is the attention-based sequence-to-sequence decoder of the ASR model.
    Its purpose is to take the encoded speech representation and generate the phoneme sequence 
    (transcription) in an autoregressive manner, while also producing an attention alignment 
    between the audio and the text. The architecture of ASRS2S is reminiscent of decoder 
    networks in sequence-to-sequence models (like the decoder in Tacotron 2 or Listen-Attend-
    Spell), featuring an embedding layer, an RNN, and an attention mechanism:
    """
    def __init__(self,
                 embedding_dim=256,
                 hidden_dim=512,
                 n_location_filters=32,
                 location_kernel_size=63,
                 n_token=40):
        """
        The `self.attention_layer` learns the alignment between audio frames and phoneme outputs.
        Because it’s location-aware, it helps enforce monotonic progression (it tends to move
        forward through the audio as output progresses, rather than jumping around).
        """
        super().__init__()
        
        self.embedding = nnx.Embed(n_token, 
                                   embedding_dim, 
                                   embedding_init=nnx.initializers.uniform(scale=math.sqrt(6 / hidden_dim)),
                                   rngs=nnx.Rngs(0))

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nnx.Linear(self.decoder_rnn_dim, n_token, rngs=nnx.Rngs(0))

        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size
        )

        # Joint hidden representation
        self.decoder_rnn = nnx.nn.recurrent.OptimizedLSTMCell(
            self.decoder_rnn_dim + embedding_dim,
            self.decoder_rnn_dim,
            rngs=nnx.Rngs(0)
        )
        
        self.project_to_hidden = LinearNorm(self.decoder_rnn_dim*2, hidden_dim)

        self.dropout = nnx.Dropout(rate=0.5, rngs=nnx.Rngs(0))
        self.sos = 1
        self.eos = 2

    def initialize_decoder_states(self, memory, mask):
        """
        memory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        """
        B, L, H = memory.shape
        self.decoder_hidden = jnp.zeros((B, self.decoder_rnn_dim))
        self.decoder_cell = jnp.zeros((B, self.decoder_rnn_dim))
        self.attention_weights = jnp.zeros((B, L))
        self.attention_weights_cum = jnp.zeros((B, L))
        self.attention_context = jnp.zeros((B, H))
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        self.unk_index = 3
        self.random_mask = 0.1

    def __call__(self, memory, memory_mask, text_input):
        """
        memory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        memory.shape = (B, L, )
        texts_input.shape = (B, T)
        """
        self.initialize_decoder_states(memory, memory_mask)
        # text random mask
        key = jax.random.PRNGKey(0)  # You should use a proper key management system
        random_mask = jax.random.uniform(key, shape=text_input.shape) < self.random_mask
        _text_input = text_input.copy()
        _text_input = jnp.where(random_mask, self.unk_index, _text_input)
        
        # Embedding
        decoder_inputs = self.embedding(_text_input)  # [B, T, channel]
        decoder_inputs = jnp.transpose(decoder_inputs, (1, 0, 2))  # -> [T, B, channel]
        
        # Create start embedding and concat
        batch_size = decoder_inputs.shape[1]
        start_tokens = jnp.full((batch_size,), self.sos)
        start_embedding = self.embedding(start_tokens)  # [B, channel]
        start_embedding = jnp.expand_dims(start_embedding, axis=0)  # [1, B, channel]
        decoder_inputs = jnp.concatenate([start_embedding, decoder_inputs], axis=0)
        
        # Initialize output containers (though typically in JAX you'd return these values rather than append to lists)
        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.shape[0]:
            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]
        hidden_outputs, logit_outputs, alignments = self.parse_decoder_outputs(hidden_outputs, logit_outputs, alignments)
        
        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input):
        cell_input = jnp.concatenate((decoder_input, self.attention_context), axis=-1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input,
            (self.decoder_hidden, self.decoder_cell)
        )
        attention_weights_cat = jnp.concatenate(
        (jnp.expand_dims(self.attention_weights, axis=1), # axis=1 is equivalent to dim=1 in torch
            jnp.expand_dims(self.attention_weights_cum, axis=1)),
            axis=1 # axis=1 is equivalent to dim=1 in torch
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask
        )
        self.attention_weights_cum += self.attention_weights

        hidden_and_context = jnp.concatenate((self.decoder_hidden, self.attention_context), axis=-1)
        hidden = nnx.tanh(self.project_to_hidden(hidden_and_context))

        # dropout to increasing g
        logit = self.project_to_n_symbols(
            self.dropout(hidden)
        )

        return hidden, logit, self.attention_weights
    
    def parse_decoder_outputs(self, hidden, logit, alignments):
        # For JAX, we need to specify all dimensions when using transpose
        # -> [B, T_out + 1, max_time]
        alignments = jnp.stack(alignments)
        alignments = jnp.transpose(alignments, (1, 0, 2))  # From [T, B, max_time] to [B, T, max_time]
        
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1, n_symbols]
        logit = jnp.stack(logit)
        logit = jnp.transpose(logit, (1, 0, 2))  # From [T, B, n_symbols] to [B, T, n_symbols]
        
        # [T_out + 1, B, hidden_dim] -> [B, T_out + 1, hidden_dim]
        hidden = jnp.stack(hidden)
        hidden = jnp.transpose(hidden, (1, 0, 2))  # From [T, B, hidden_dim] to [B, T, hidden_dim]
        
        return hidden, logit, alignments
    
if __name__ == "__main__":
    model = LinearNorm(512,40)
    nnx.display(model)