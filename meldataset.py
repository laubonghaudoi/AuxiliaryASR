# coding: utf-8

import logging
import os.path as osp
import random
from typing import List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from g2p_en import G2p
from torch.utils.data import DataLoader

from text_utils import TextCleaner
from utils import get_data_path_list

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}


class MelDataset(torch.utils.data.Dataset):
    """
    Args:
        data_list: list of strings generated from `get_data_path_list`,
                each string is formatted in `wave_path|text|speaker_id`.
        dict_path: path to the dictionary file.
        sr: sample rate of the wave files.

    Returns:
        wave_tensor (torch.FloatTensor): wave data tensor.
        acoustic_feature (torch.FloatTensor): mel spectrogram tensor.
        text_tensor (torch.LongTensor): text tensor.
        speaker_id (int): speaker id.
    """

    def __init__(self,
                 data_list: List[str],
                 dict_path: str = DEFAULT_DICT_PATH,
                 sr: int = 24000
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]

        # list of [wave_path, text, speaker_id]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4

        self.g2p = None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        If the phonemes is longer than 1/3 of the time-dimensio of the mel
        spectrogram. If it is, it uses interpolation to increase the size
        of the mel spectrogram to match the length of the text. The intuition
        behind this is that the phonemes should be longer than the mel spectrogram
        to capture the temporal information of the speech signal.

        Returns:
            wave_tensor (torch.FloatTensor): wave data tensor.
            acoustic_feature (torch.FloatTensor): mel spectrogram tensor.
            phonemes_tensor (torch.LongTensor): text tensor.
            path (str): path to the wave file.
        """
        # wave_path, text, speaker_id
        data = self.data_list[idx]
        # wave is a one-dimensional numpy array
        wave, phoneme_indices, speaker_id = self._load_raw(data)
        # Shape [time,]
        phonemes_tensor = torch.LongTensor(phoneme_indices)

        # Convert wave to tensor
        wave_tensor = torch.from_numpy(wave).float()
        # Convert wave to mel spectrogram, shape [n_mels, time]
        mel_tensor = self.to_melspec(wave_tensor)

        # If the phonemes tensor is longer than 1/3 of the mel spectrogram, interpolate
        if (phonemes_tensor.size(0) + 1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(phonemes_tensor.size(0) + 1) * 3, align_corners=False,
                mode='linear').squeeze(0)

        # [n_mels, time]
        acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

        # Make sure the length of the mel spectrogram is even (I don't know why this is necessary)
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, phonemes_tensor, data[0]

    def _load_raw(self, data: List[str]):
        """
        Given a list of waveform file path, text and speaker id, 
        return the waveform tensor, text tensor and speaker id.

        Args:
            data: list of [wave_path, text, speaker_id]

        Returns:
            wave: waveform numpy array.
            phoneme_indices: phoneme index tensor.
            speaker_id: speaker id.
        """
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        # get wave form
        wave, sr = sf.read(wave_path)

        # phonemize the text
        if self.g2p is None:
            self.g2p = G2p()
        phonemes = self.g2p(text.replace('-', ' '))
        if "'" in phonemes:
            phonemes.remove("'")
        phoneme_indices = self.text_cleaner(phonemes)

        blank_index = self.text_cleaner.word_index_dictionary[" "]
        phoneme_indices.insert(0, blank_index)  # add a blank at the beginning (silence)
        phoneme_indices.append(blank_index)  # add a blank at the end (silence)

        return wave, phoneme_indices, speaker_id


class Collater(object):
    """
    Args:
        return_wave (bool): if true, will return the wave data along with spectrogram. 

    Returns:
        phonemes (torch.LongTensor): phoneme tensor.
        input_lengths (torch.LongTensor): phoneme tensor length.
        mels (torch.FloatTensor): mel spectrogram tensor.
        output_lengths (torch.LongTensor): mel spectrogram tensor length.
        (optional) paths (List[str]): list of paths to the wave files.
        (optional) waves (List[np.ndarray]): list of wave data.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]

        # returns the indices that would sort the lengths array in ascending order.
        batch_indexes = np.argsort(lengths)[::-1]
        # reorders the batch list based on batch_indexes. Now, the samples with the longest mel spectrograms will be at the beginning of the batch
        # Why sorting: Sorts the batch by the length of the spectrograms (descending). This is a common trick to improve efficiency when training with padded sequences. It reduces the amount of padding added.
        batch = [batch[bid] for bid in batch_indexes]
        # Gets the number of mel frequency bins (which is the same for all samples). batch[0][1] is the mel spectrogram of the first sample, and size(0) is the number of rows (mel bands).
        nmels = batch[0][1].size(0)
        # Calculates the length of the longest mel spectrogram in the batch (time dimension)
        max_mel_length = max([b[1].shape[1] for b in batch])
        # Calculates the length of the longest phoneme sequence in the batch
        max_phoneme_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        phonemes = torch.zeros((batch_size, max_phoneme_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()

        paths = ['' for _ in range(batch_size)]
        # Each sample in this batch
        for bid, (_, mel, phoneme, path) in enumerate(batch):
            mel_size = mel.size(1)
            phoneme_size = phoneme.size(0)

            # Copies the current mel spectrogram into the mels tensor.
            # It only copies up to mel_size, and the rest of the row will stay as zero, this is the padding process.
            mels[bid, :, :mel_size] = mel
            # Copies the current phoneme sequence into the phonemes tensor.
            # The rest of the row will stay as zero
            phonemes[bid, :phoneme_size] = phoneme

            input_lengths[bid] = phoneme_size
            output_lengths[bid] = mel_size

            paths[bid] = path

            # phonemes must be shorter than the acoustic feature.
            assert (phoneme_size < (mel_size // 2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return phonemes, input_lengths, mels, output_lengths, paths, waves

        return phonemes, input_lengths, mels, output_lengths


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader


if __name__ == "__main__":
    train_list, val_list = get_data_path_list("Data/train_list.txt", "Data/val_list.txt")
    mel_dataset = build_dataloader(train_list, batch_size=1)
    for batch in mel_dataset:
        pass
