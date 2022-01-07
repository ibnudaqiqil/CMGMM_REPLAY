import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np


class UrbanSoundDataset(Dataset):
    #rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])

        self.file_path = file_path
        # UrbanSound8K uses two channels, this will convert them to one
        #self.mixer = torchaudio.transforms.DownmixMono()
        self.folderList = folderList

    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path + "fold" + \
            str(self.folders[index]) + "/" + self.file_names[index]
        sound = torchaudio.load(path, out=None, normalization=True)
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
       # soundData = self.mixer(sound[0])
        soundData = torch.mean(sound, dim=0, keepdim=True)
        #downsample the audio to ~8kHz
        # tempData accounts for audio clips that are too short
        tempData = torch.zeros([160000, 1])
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]

        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        # take every fifth sample of soundData
        soundFormatted[:32000] = soundData[::5]
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)
