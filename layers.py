import random

import torch
import torch.nn.functional as F
import torchaudio.functional as audio_F
from torch import nn

random.seed(0)


def _get_activation_fn(activ):
    """Return the activation function corresponding to the given name.

    Parameters:
        activ (str): The name of the activation function. Supported values are 'relu', 'lrelu', and 'swish'.

    Returns:
        torch.nn.Module or callable: The activation function.
    """
    if activ == 'relu':
        return nn.ReLU()
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif activ == 'swish':
        return lambda x: x * torch.sigmoid(x)
    else:
        raise RuntimeError('Unexpected activ type %s, expected [relu, lrelu, swish]' % activ)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        """Initialize LinearNorm layer.

        Parameters:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            bias (bool, optional): Whether to include a bias term. Default is True.
            w_init_gain (str, optional): Gain factor used for Xavier uniform initialization. Default is 'linear'.
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """Forward pass of LinearNorm.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.
        """
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        """Initialize ConvNorm layer.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 1.
            stride (int, optional): Stride for the convolution. Default is 1.
            padding (int, optional): Padding value; if None, computed automatically assuming odd kernel.
            dilation (int, optional): Dilation rate for the convolution. Default is 1.
            bias (bool, optional): If True, adds a bias term. Default is True.
            w_init_gain (str, optional): Gain factor for Xavier uniform initialization. Default is 'linear'.
            param (Any, optional): Additional parameter for initialization.
        """
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, signal):
        """Apply a 1D convolution using ConvNorm layer.

        Parameters:
            signal (torch.Tensor): Input tensor of shape (batch, channels, length).

        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        conv_signal = self.conv(signal)
        return conv_signal


class CausualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=True, w_init_gain='linear', param=None):
        """Initialize CausualConv layer with causal padding adjustment.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel.
            stride (int, optional): Convolution stride.
            padding (int, optional): Base padding value before doubling.
            dilation (int, optional): Dilation rate.
            bias (bool, optional): Whether to use bias.
            w_init_gain (str, optional): Gain for Xavier initialization.
            param (Any, optional): Additional parameter for Xavier calculation.
        """
        super(CausualConv, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding,
                              dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        """Apply causal 1D convolution and adjust output tensor by removing excess padded values.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch, channels, length).

        Returns:
            torch.Tensor: Output tensor after causal convolution.
        """
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x


class CausualBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        """Initialize CausualBlock.

        Parameters:
            hidden_dim (int): Number of channels for input and output.
            n_conv (int, optional): Number of convolutional layers in the block.
            dropout_p (float, optional): Dropout probability. Default is 0.2.
            activ (str, optional): Activation function name. Default is 'lrelu'.
        """
        super(CausualBlock, self).__init__()
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])

    def forward(self, x):
        """Apply the CausualBlock to the input tensor with residual connections.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through causal convolution blocks.
        """
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='lrelu', dropout_p=0.2):
        """Create a causal convolution sub-block for the CausualBlock.

        Parameters:
            hidden_dim (int): Number of channels.
            dilation (int): Dilation factor for the convolution.
            activ (str, optional): Activation function name.
            dropout_p (float, optional): Dropout probability.

        Returns:
            nn.Sequential: A sequential block combining causal convolutions, activation, batch normalization, and dropout.
        """
        layers = [
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_p),
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='relu'):
        """Initialize ConvBlock.

        Parameters:
            hidden_dim (int): Number of channels for the convolution layers.
            n_conv (int, optional): Number of convolution layers in the block.
            dropout_p (float, optional): Dropout probability. Default is 0.2.
            activ (str, optional): Activation function name. Default is 'relu'.
        """
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])

    def forward(self, x):
        """Apply the ConvBlock to the input tensor with residual connections.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through convolution blocks.
        """
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='relu', dropout_p=0.2):
        """Create a convolution sub-block for ConvBlock.

        Parameters:
            hidden_dim (int): Number of channels.
            dilation (int): Dilation rate for the convolution.
            activ (str, optional): Activation function name.
            dropout_p (float, optional): Dropout probability.

        Returns:
            nn.Sequential: A sequential convolution block with normalization, activation, and dropout.
        """
        layers = [
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        """Initialize LocationLayer.

        Parameters:
            attention_n_filters (int): Number of filters for the convolution.
            attention_kernel_size (int): Size of the convolution kernel.
            attention_dim (int): Dimension of the output attention features.
        """
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        """Process attention weights using convolution and a dense layer.

        Parameters:
            attention_weights_cat (torch.Tensor): Concatenated attention weights of shape (B, 2, T).

        Returns:
            torch.Tensor: Processed attention features of shape (B, T, attention_dim).
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        """Initialize Attention layer.

        Parameters:
            attention_rnn_dim (int): Dimension of the attention RNN hidden state.
            embedding_dim (int): Dimension of the encoder outputs.
            attention_dim (int): Dimension for attention computations.
            attention_location_n_filters (int): Number of filters for location-based attention.
            attention_location_kernel_size (int): Kernel size for the location convolution.
        """
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
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
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        """Initialize ForwardAttentionV2 layer.

        Parameters:
            attention_rnn_dim (int): Dimension of the attention RNN hidden state.
            embedding_dim (int): Dimension of the encoder outputs.
            attention_dim (int): Dimension used for attention.
            attention_location_n_filters (int): Number of filters for location-based attention.
            attention_location_kernel_size (int): Kernel size for the location convolution.
        """
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
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
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """Perform a forward pass for the ForwardAttentionV2 layer.

        Parameters:
            attention_hidden_state (torch.Tensor): The hidden state from the attention RNN.
            memory (torch.Tensor): Encoder outputs (B, T, D).
            processed_memory (torch.Tensor): Processed encoder outputs for attention (B, T, attention_dim).
            attention_weights_cat (torch.Tensor): Concatenated previous and cumulative attention weights (B, 2, T).
            mask (torch.Tensor): Binary mask for padded data.
            log_alpha (torch.Tensor): Log probability of previous attention distribution (B, T).

        Returns:
            tuple: (attention_context, attention_weights, log_alpha_new) where:
                attention_context (torch.Tensor): Attention context vector.
                attention_weights (torch.Tensor): Attention weight distribution.
                log_alpha_new (torch.Tensor): Updated log probabilities for attention.
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        # log_energy =

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        # attention_weights = F.softmax(alignment, dim=1)

        # content_score = log_energy.unsqueeze(1) #[B, MAX_TIME] -> [B, 1, MAX_TIME]
        # log_alpha = log_alpha.unsqueeze(2) #[B, MAX_TIME] -> [B, MAX_TIME, 1]

        # log_total_score = log_alpha + content_score

        # previous_attention_weights = attention_weights_cat[:,0,:]

        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        for sft in range(2):
            shifted = log_alpha[:, :max_time - sft]
            shift_padded = F.pad(shifted, (sft, 0), 'constant', self.score_mask_value)
            log_alpha_shift_padded.append(shift_padded.unsqueeze(2))

        biased = torch.logsumexp(torch.cat(log_alpha_shift_padded, 2), 2)

        log_alpha_new = biased + log_energy

        attention_weights = F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new


class PhaseShuffle2d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle2d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        """Shuffle the phase of a 2D tensor along the last dimension in a circular manner.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, M, L).
            move (int, optional): Number of positions to shift. If None, a random shift between -n and n is chosen.

        Returns:
            torch.Tensor: Tensor with phase-shuffled data.
        """
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = torch.cat([right, left], dim=3)
        return shuffled


class PhaseShuffle1d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle1d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        """Shuffle the phase of a 1D tensor along the last dimension in a circular manner.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, L).
            move (int, optional): Number of positions to shift. If None, a random shift between -n and n is chosen.

        Returns:
            torch.Tensor: Tensor with phase-shuffled data.
        """
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :move]
            right = x[:, :, move:]
            shuffled = torch.cat([right, left], dim=2)

        return shuffled


class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        """Initialize MFCC layer for computing Mel Frequency Cepstral Coefficients.

        Parameters:
            n_mfcc (int, optional): Number of MFCC coefficients to compute. Default is 40.
            n_mels (int, optional): Number of Mel filter banks. Default is 80.

        Note:
            Creates a DCT (Discrete Cosine Transform) matrix based on the specified parameters.
        """
        super(MFCC, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_specgram):
        """Compute MFCC features from a Mel spectrogram.

        Parameters:
            mel_specgram (torch.Tensor): Input Mel spectrogram of shape (B, n_mels, time) or (n_mels, time).

        Returns:
            torch.Tensor: MFCC features with shape (B, n_mfcc, time) if batched, otherwise (n_mfcc, time).
        """
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)

        # unpack batch
        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc
