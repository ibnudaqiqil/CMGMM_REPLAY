import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

"""
Class to convert Stereo Audio to Mono
"""


class ToMono:
    def __init__(self, channel_first=True):
        self.channel_first = channel_first

    def __call__(self, x):
        assert len(x.shape) == 2, "Can only take two dimenshional Audio Tensors"
        output = torch.mean(
            x, dim=0) if self.channel_first else torch.mean(x, dim=1)
        return output
        
#dataset for DCAE
class DCaseDataset(Dataset):
  
    labelind2name = {
        0: "beach",
        1: "bus",
        2: "cafe/restaurant",
        3: "car",
        4: "city_center",
        5: "forest_path",
        6: "grocery_store",
        7: "home",
        8: "library",
        9: "metro_station",
        10: "office",
        11: "park",
        12: "residential_area",
        13: "train",
        14: "tram",
    }
    name2labelind = {
        "beach": 0,
        "bus": 1,
        "cafe/restaurant": 2,
        "car": 3,
        "city_center": 4,
        "forest_path": 5,
        "grocery_store": 6,
        "home": 7,
        "library": 8,
        "metro_station":9,
        "office" :10,
        "park":11,
        "residential_area":12,
        "train":13,
        "tram":14

    }

    def __init__(self, root_dir, split, extension="txt"):
        """

        :param root_dir:
        :param split:
        """

        # Open csv files
        self.ext = extension
        self.split = split
        self.root_dir = root_dir
        if split == "train":
            csv_path = root_dir + "/evaluation_setup/fold1_train."+self.ext
            meta_path = root_dir + "/meta."+self.ext
        elif split == "val":
            csv_path = root_dir + "/evaluation_setup/fold1_evaluate."+self.ext
            meta_path = root_dir + "/meta."+self.ext
        elif split == "test":
            csv_path = root_dir + "/evaluation_setup/fold1_test."+self.ext
            meta_path = None
        else:
            raise ValueError("Split not implemented")
        csvData = pd.read_csv(csv_path, sep="\t")
        metaData = pd.read_csv(
            meta_path, sep="\t") if meta_path is not None else None

        # In test mode, just get file list
        if split == "test":
            self.file_names = []
            for i in range(0, len(csvData)):
                self.file_names.append(csvData.iloc[i, 0])
            return

        # Lists of file names and labels
        self.file_names, self.labels = [], []
        for i in range(0, len(csvData)):
            self.file_names.append(csvData.iloc[i, 0])
            self.labels.append(csvData.iloc[i, 1])

        # Device for each audio file
        #self.devices = {}
        #for i in range(0, len(metaData)):
        #    self.devices[metaData.iloc[i, 0]] = metaData.iloc[i, 3]

        # Transform class name to index
        self.labels = [self.name2labelind[name] for name in self.labels]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        # Load data
        filepath = self.root_dir + self.file_names[index]
        sound, sfreq = torchaudio.load(filepath)
        assert sound.shape[0] == 1, "Expected mono channel"
        sound = torch.mean(sound, dim=0)
        assert sfreq == 44100, "Expected sampling rate of 44.1 kHz"
    
        # Remove last samples if longer than expected
        if sound.shape[-1] >= 441000:
            sound = sound[:441000]

        if self.split == "test":
            return sound, 255, self.file_names[index], "unknown"
        else:
            return (
                sound,
                self.labels[index],
                self.file_names[index],
               # self.devices[self.file_names[index]],
            )

    def __len__(self):
        return len(self.file_names)
