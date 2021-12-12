import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import os
from torchvision.transforms import ToTensor
import numpy as np
from torchsummary import summary

class Dcase2013(Dataset):

  def __init__(self, path, class_map):
    super().__init__()
    self.dataset_path = path
    self.class_map = class_map
    self.sample_rate = 22050
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.n_sample = 661500  # 30 sec of each audio
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=96).to(self.device)

  def __len__(self):
    return len(os.listdir(self.dataset_path))

  def __getitem__(self, index):
    file_name = os.listdir(self.dataset_path)[index]
    class_name = file_name.split('0')[0].split('1')[0]
    label = self.class_map.index(class_name)
    file_path = os.path.join(self.dataset_path, file_name)
    signal, sample_rate = torchaudio.load(filepath=file_path)
    signal = signal.to(self.device)
    #resample if necessary
    if(sample_rate != self.sample_rate):
      resampler = torchaudio.transforms.Resample(
          sample_rate, self.sample_rate).to(self.device)
      signal = resampler(signal)
    # stereo to mono convert
    if(signal.shape[0] > 1):
      signal = torch.mean(signal, dim=0, keepdim=True)
    #adjust lenght

    #cut if necessary
    if(signal.shape[1] > self.n_sample):
      signal = signal[:, :self.n_sample]
    #pad if necessary
    elif(signal.shape[1] < self.n_sample):
      signal = nn.functional.pad(
          signal, (0, self.n_sample-signal.shape[1]))  # right pad at last dim
    else:
      pass

    signal = self.mel_spectrogram(signal)
    return signal, label
