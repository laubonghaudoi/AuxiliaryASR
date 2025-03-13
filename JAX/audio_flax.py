import jax
import jax.numpy as jnp
from jax.scipy.signal import stft
import torch
import torchaudio
import numpy as np

class MelSpectrogram:
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: int = None,
                 hop_length: int = None,
                 f_min: float = 0.0,
                 f_max: float = None,
                 n_mels: int = 128,
                 window_fn=None,   # e.g. jax.numpy.hanning or string like 'hann'
                 power: float = 2.0,
                 normalized: bool = False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.f_min = f_min
        # If f_max not provided, use Nyquist (sample_rate/2)
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.n_mels = n_mels
        self.power = power
        self.normalized = normalized

        # Prepare the window function
        if window_fn is None:
            # Default to Hann window (periodic Hann)
            window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(self.win_length) / self.win_length)
        elif isinstance(window_fn, str):
            # Use JAX scipy get_window if available, otherwise handle common names
            if window_fn.lower() in ["hann", "hann_window", "hanning"]:
                window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(self.win_length) / self.win_length)
            elif window_fn.lower() in ["hamming"]:
                window = jnp.hamming(self.win_length)
            else:
                raise ValueError(f"Unsupported window type: {window_fn}")
        elif callable(window_fn):
            # Assume window_fn returns an array when called with win_length
            w = window_fn(self.win_length)
            window = jnp.array(w) if not isinstance(w, jnp.ndarray) else w
        else:
            raise ValueError("window_fn must be None, a string, or a callable generating a window.")
        # If win_length < n_fft, zero-pad the window to n_fft (as PyTorch does)
        if self.win_length < self.n_fft:
            pad_width = self.n_fft - self.win_length
            # Pad on both sides evenly (or one extra on right if odd difference)
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            window = jnp.pad(window, (left_pad, right_pad), mode='constant', constant_values=0.0)
        self.window = window  # store the window (JAX array)

        # Precompute the Mel filter bank matrix (shape: n_mels x (n_fft//2 + 1))
        # 1. Compute Mel scale values for f_min and f_max
        def hz_to_mel(f):  # HTK mel scale
            return 2595.0 * jnp.log10(1.0 + f / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10**(mel / 2595.0) - 1.0)
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        # 2. Linear space in Mel domain, then convert to Hz
        mel_points = jnp.linspace(mel_min, mel_max, self.n_mels + 2)
        freq_points = mel_to_hz(mel_points)
        # 3. Get the discrete Fourier frequencies for each FFT bin (one-sided)
        n_freqs = self.n_fft // 2 + 1
        fft_bin_frequencies = jnp.linspace(0.0, self.sample_rate / 2.0, num=n_freqs)
        # 4. Initialize mel filter bank matrix
        # We will compute the filter weights for each mel band with vectorized operations.
        # Create arrays for the start, center, end frequencies of each mel filter
        start_freqs = freq_points[:-2]   # length n_mels (f(m-1))
        center_freqs = freq_points[1:-1]  # length n_mels (f(m))
        end_freqs = freq_points[2:]     # length n_mels (f(m+1))
        # Expand dims for broadcasting: mel_filters shape (n_mels, n_freqs)
        start_freqs = start_freqs[:, None]  # shape (n_mels, 1)
        center_freqs = center_freqs[:, None]
        end_freqs = end_freqs[:, None]
        # Expand fft_bin_frequencies to shape (1, n_freqs) for broadcasting
        freq_grid = fft_bin_frequencies[None, :]  # shape (1, n_freqs)

        # Compute rising and falling slopes for all filters in one go
        # Slope up: (freq - start) / (center - start) for freq between start and center
        rise = (freq_grid - start_freqs) / (center_freqs - start_freqs + 1e-10)
        # Slope down: (end - freq) / (end - center) for freq between center and end
        fall = (end_freqs - freq_grid) / (end_freqs - center_freqs + 1e-10)
        # Use jnp.clip or where to apply conditions:
        # If freq < start or freq > end, weight = 0.
        # If start <= freq <= center, use rising slope; if center <= freq <= end, use falling slope.
        mel_filters = jnp.where(freq_grid < start_freqs, 0.0, 
                                 jnp.where(freq_grid <= center_freqs, rise,
                                           jnp.where(freq_grid <= end_freqs, fall, 0.0)))
        # For numerical stability, clip to [0,1]
        mel_filters = jnp.clip(mel_filters, 0.0, 1.0)
        # Optionally, implement Slaney normalization if needed (not requested here).
        # mel_filters norm ('slaney') would divide each filter by the trapezoid area (end-start).
        self.mel_filter = mel_filters  # shape (n_mels, n_freqs)

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Mel spectrogram of the input waveform.
        Expected waveform shape: (..., time), returns array of shape (..., n_mels, time_frames).
        """
        # 1. Compute STFT. jax.scipy.signal.stft returns (freqs, times, Zxx)
        freqs, times, Zxx = stft(waveform, 
                                            fs=self.sample_rate,
                                            window=self.window,
                                            nperseg=self.win_length,
                                            noverlap=self.win_length - self.hop_length,
                                            nfft=self.n_fft,
                                            boundary=None,  # no padding at edges (can use 'zeros')
                                            padded=False,    # pad end of signal
                                            return_onesided=True) 
        # Zxx shape: (..., n_freqs, time_frames), complex dtype
        spectrogram = Zxx
        # 2. Normalize STFT if needed (scale by 1/sqrt(win_length))
        if self.normalized:
            spectrogram = spectrogram * (1.0 / jnp.sqrt(self.win_length))
        # 3. Convert to magnitude (power) spectrogram
        # If power==1, magnitude; if 2, power; if other, fractional power of magnitude.
        if self.power is not None:
            spectrogram = jnp.abs(spectrogram) ** self.power  # result is real
        else:
            # If power is None, we keep complex STFT (not typical for MelSpectrogram)
            spectrogram = jnp.abs(spectrogram)  # default to magnitude
        # 4. Apply Mel filter bank. We need to multiply along the frequency axis.
        # Use einsum to handle any leading batch dimensions:
        mel_spec = jnp.einsum('mf,...ft->...mt', self.mel_filter, spectrogram)
        # mel_spec shape: (..., n_mels, time_frames)
        return mel_spec

if __name__ == "__main__":
    # Define the same Hann window:
    def hann_window_400():
        n = jnp.arange(400)
        # periodic Hann
        return 0.5 - 0.5*jnp.cos(2.0*jnp.pi*n/400.0)
    
    jax_melspec = MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=200,
    f_min=0.0,
    f_max=8000.0,
    n_mels=64,
    window_fn="hann",    # Default Hann window
    power=2.0,
    normalized=False,
    )

    torch_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,          # match n_fft
    hop_length=200,          # 50%  hop
    f_min=0.0,
    f_max=8000.0,
    n_mels=64,
    window_fn=torch.hann_window,
    power=2.0,
    normalized=False,
    center=False,            # disable center padding
    pad_mode='constant'      # match padding
)
    
    # Make reproducible random data
    torch.manual_seed(0)

    # Single-channel, 1-second of audio at 16kHz
    waveform_torch = torch.randn(16000)  # shape: (time,)

    # Convert the PyTorch Tensor to a JAX array
    waveform_jax = jnp.array(waveform_torch.numpy())
    # PyTorch mel spectrogram
    mel_torch = torch_melspec(waveform_torch)
    # shape: (channel=1, n_mels=64, time_frames)

    # JAX mel spectrogram
    mel_jax = jax_melspec(waveform_jax) 
    # shape: (n_mels=64, time_frames)

    print("PyTorch mel_spec shape:", tuple(mel_torch.shape))
    print("JAX mel_spec shape:", mel_jax.shape)

    # Remove the channel dimension for direct comparison
    mel_torch_np = mel_torch.squeeze(0).numpy()  # shape: (64, time_frames)
    mel_jax_np = np.array(mel_jax)               # shape: (64, time_frames)

    # Mean Absolute Error
    mae = np.mean(np.abs(mel_torch_np - mel_jax_np))
    max_diff = np.max(np.abs(mel_torch_np - mel_jax_np))

    print("Mean absolute difference:", mae)
    print("Max absolute difference:", max_diff)