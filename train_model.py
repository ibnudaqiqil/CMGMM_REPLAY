import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import os
from torchvision.transforms import ToTensor
import numpy as np
from torchsummary import summary
import csv
import pandas as pd
from sklearn import preprocessing
class_map = ["beach", "bus", "cafe/restaurant", "car", "city_center", "forest_path",
             "grocery_store", "home", "library", "metro_station", "office", "park", 'residential_area', "train", "tram"]
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1
LEARNING_RATE = 0.0002

def train_one_epoch(model, dataloader, optimizer, loss_f, device):

  for signal, label in dataloader:
    signal, label = signal.to(device), label.to(device)
    output = model(signal)
    loss = loss_f(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"loss after this epoch: {loss.item()}")


def train(model, dataloader, optimizer, loss_f, device, epochs):
  model.train()
  for i in range(epochs):
    print(f" Epoch: {i+1}")
    train_one_epoch(model, dataloader, optimizer, loss_f, device)
    print("----------------------")


class CNNx(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.flat = nn.Flatten()
    self.linear = nn.Linear(
        in_features=128*7*82,
        out_features=15
    )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flat(x)
    x = self.linear(x)
    predict = self.softmax(x)
    return predict


class Dcase2013(Dataset):

    def __init__(self, data_path="data",  name='dataset'):
        super().__init__()
        # Folder name for dataset
        self.name = name

        # Path to the dataset
        self.local_path = os.path.join(data_path, self.name)
       # self.dataset_path = os.path.join(self.local_path, 'audio')
        # Evaluation setup folder
        self.evaluation_setup_folder = 'evaluation_setup'

        # Path to the folder containing evaluation setup files
        self.evaluation_setup_path = os.path.join(
                self.local_path, self.evaluation_setup_folder)

        # Meta data file, csv-format
        self.meta_filename = 'meta.txt'

        # Path to meta data file
        self.meta_file = os.path.join(self.local_path, self.meta_filename)

        # Error meta data file, csv-format
        self.error_meta_filename = 'error.txt'

        # Path to error meta data file
        self.error_meta_file = os.path.join(
                self.local_path, self.error_meta_filename)

        # Hash file to detect removed or added files
        self.filelisthash_filename = 'filelist.python.hash'

        # Number of evaluation folds
        self.evaluation_folds = 1

            # List containing dataset package items
            # Define this in the inherited class.
            # Format:
            # {
            #        'remote_package': download_url,
            #        'local_package': os.path.join(self.local_path, 'name_of_downloaded_package'),
            #        'local_audio_path': os.path.join(self.local_path, 'name_of_folder_containing_audio_files'),
            # }
        #self.package_list = {'local_package': os.path.join(self.local_path, 'audio') }

        # List of audio files
        self.files = None

        # List of meta data dict
        self.meta_data = None

        # List of audio error meta data dict
        self.error_meta_data = None

        # Training meta data for folds
        self.evaluation_data_train = {}

        # Testing meta data for folds
        self.evaluation_data_test = {}

        # Recognized audio extensions
        self.audio_extensions = {'wav', 'flac'}
       
        #self.data = pd.read_table(self.meta_file, sep='\t', header=None)
        data = np.loadtxt(self.meta_file, delimiter='\t', dtype=str)
        self.x = data[:,0]
        le = preprocessing.LabelEncoder()

        targets = le.fit_transform(data[:, 1])
        self.y = torch.from_numpy(targets)
        self.sample_rate = 22050
       
        
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_sample = 661500  # 30 sec of each audio
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=96).to(self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        #file_name = os.listdir(self.dataset_path)[index]
        #class_name = file_name.split('0')[0].split('1')[0]
        #//label = self.class_map.index(class_name)
        file_path = os.path.join(self.local_path, self.x[index])
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
        return signal, self.y[index]

model = CNNx()
model.to(DEVICE)
summary(model, (1, 96, 1292))


dcase13 = Dcase2013( name="TUT-acoustic-scenes-2016-development")
#d1 = dcase13[0];
signal=0
if(signal==1):
    dcase13.load_state_dict(torch.load("model.pth"))
    dcase13.eval()
else:
    dataloader = DataLoader(dcase13, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_f = nn.CrossEntropyLoss()
    print(f"we are using {DEVICE}")
    train(model, dataloader, optimizer, loss_f, DEVICE, EPOCHS)
    torch.save(model.state_dict(), "model.pth")

dcase13_test = Dcase2013(name="TUT-acoustic-scenes-2016-evaluation")
dataloader_test = DataLoader(dcase13_test, batch_size=BATCH_SIZE, shuffle=True)

correct = 0
total = 100
for input, label in dataloader_test:
  batchlabel = label
  batchpredict = dcase13(input)
  for i in range(len(batchlabel)):
    if(batchpredict[i].item() == batchlabel[i].item()):
      correct += 1

print(f" Acurracy after {EPOCHS} epoch : {correct/total}")
