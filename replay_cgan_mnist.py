from torchvision.utils import save_image
from models.CGAN import Generator, Discriminator, train_replayer, weights_init_normal, sample_image
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import models.GAN as models
import helper.LIBS as lib
from torch.utils.tensorboard import SummaryWriter
from datasets.MNIST import MNIST_IncrementalDataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


writer = SummaryWriter()
batch_size = num_noise = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

TrainDataSet = []
TestDataSet = []

# tasks to use
task_classes_arr = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
tasks_num = len(task_classes_arr)  # 5

for i in range(tasks_num):
    # MNIST dataset
    _TrainDataSet = MNIST_IncrementalDataset(source='./data/',
                                            train=True,
                                            transform=transform,
                                            download=True,
                                            classes=range(i * 2, (i+1) * 2))

    _TestDataSet = MNIST_IncrementalDataset(source='./data/',
                                           train=False,
                                           transform=transform,
                                           download=True,
                                           classes=range(0, (i+1) * 2))
    TrainDataSet.append(_TrainDataSet)
    TestDataSet.append(_TestDataSet)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from models.MNIST import SolverCNN



classifier = SolverCNN(L=10).to(device)
previous_generator = None
#train for every task
for task_id in range(tasks_num):
    ratio = 1 / (task_id+1)  
    if task_id > 0:
        previous_generator = 1#generator
        previous_classifier = classifier

        #lib.model_grad_switch(previous_generator, False)
        #lib.model_grad_switch(previous_classifier, False)
    print(TrainDataSet[task_id].shape)
    if(previous_generator):
        TrainDataLoaders = torch.utils.data.DataLoader(TrainDataSet[task_id],
                                                        batch_size=batch_size,
                                                        shuffle=True)
        TestDataLoaders = torch.utils.data.DataLoader(TestDataSet[task_id],
                                                    batch_size=batch_size,
                                                    shuffle=False)
    else:
        TrainDataLoaders = torch.utils.data.DataLoader(TrainDataSet[task_id],
                                                                batch_size=batch_size,
                                                                shuffle=True)
        TestDataLoaders = torch.utils.data.DataLoader(TestDataSet[task_id],
                                                        batch_size=batch_size,
                                                        shuffle=False)
    # train the generator and classifier
    print("task",task_id)
    #generator,discriminator = train_replayer(TrainDataLoaders, (task_id+1) * 2, writer)
    #img = sample_image(generator, 10, list(range(0, (task_id+1) * 2)), 100)
    #save_image(img.data, 'store/%d.png' % task_id, nrow=10, normalize=True)
            
