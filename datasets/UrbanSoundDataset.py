import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path

class UrbanSoundDataset(Dataset):
    """
    This Dataset now returns Mel Spectograms of the sound with the same sample_rate and 1 channel, instead of Waveforms
    """

    def __init__(
        self, annotations_file: Path, audio_dir: Path, transforms, target_sample_rate
    ):
        super().__init__()
        self.annotations = pd.read_csv(str(annotations_file))
        self.audio_dir = audio_dir
        self.transforms = transforms
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(
            str(audio_sample_path)
        )  # (wave_form, sample_rate)
        #print(f"sognal shape before resampling = {signal.size()}")
        signal = self._resample_if_necessary(signal, sample_rate)

        #print(f"sognal shape before mixing = {signal.size()}")
        signal = self._mix_down_if_necessary(signal)

        #print(f"sognal shape before transforming = {signal.size()}")
        #signal = self.transforms(signal)
        # tempData accounts for audio clips that are too short
        tempData = torch.zeros([160000, 1])
        if signal.numel() < 160000:
            tempData[:signal.numel()] = signal[:]
        else:
            tempData[:] = signal[:160000]

        signal = tempData
        soundFormatted = torch.zeros([32000, 1])
        # take every fifth sample of soundData
        soundFormatted[:32000] = signal[::5]
        soundFormatted = soundFormatted.permute(1, 0)

        return signal, label

    def _get_audio_sample_path(self, index):
        folder = f"fold{self.annotations.iloc[index, 5]}"
        path = self.audio_dir + "/" + folder + \
            "/" + self.annotations.iloc[index, 0]
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _resample_if_necessary(self, signal, sample_rate):
        """set the sample rate same for every datapoint in the dataset"""
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """if we have multiple channels than we have to average them on the channels dimentsion"""
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
