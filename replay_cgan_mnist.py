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
import copy
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


writer = SummaryWriter()
batch_size = num_noise = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

TrainDataSet = []
TestDataSet = []

# tasks to use
task_classes_arr = [[0, 1], 
                    [0, 1, 2, 3], 
                    [0, 1, 2, 3, 4, 5], 
                    [0, 1, 2, 3, 4, 5, 6, 7], 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
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
number_previous_concept = 6000
n_latent=100
#train for every task
for task_id in range(tasks_num):
    ratio = 1 / (task_id+1) 
    print('task_id:', task_id, 'ratio:', ratio,"-"*20)
    if task_id > 0:
        print(">>saving previous generator")
        previous_generator = copy.deepcopy(generator)
        previous_classifier = classifier

        lib.model_grad_switch(previous_generator, False)
        lib.model_grad_switch(previous_classifier, False)
    
    if(previous_generator):

        psudodata, psudodata_label = sample_image(
            previous_generator, number_previous_concept, task_classes_arr[task_id-1], n_latent)
        print("appending data from ", task_classes_arr[task_id-1])
        psudodataset = torch.utils.data.TensorDataset(
            psudodata.clone().detach(), psudodata_label.clone().detach())
        train_set = torch.utils.data.ConcatDataset([TrainDataSet[task_id], psudodataset])
        
        TrainDataLoaders = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        TestDataLoaders = torch.utils.data.DataLoader(TestDataSet[task_id],
                                                    batch_size=batch_size,
                                                    shuffle=False)
        print("train replayer using psudodata nclass= ", (task_id+1) * 2)
    else:
        TrainDataLoaders = torch.utils.data.DataLoader(TrainDataSet[task_id],
                                                                batch_size=batch_size,
                                                                shuffle=True)
        TestDataLoaders = torch.utils.data.DataLoader(TestDataSet[task_id],
                                                        batch_size=batch_size,
                                                        shuffle=False)
        print("train replayer using real nclass= ", (task_id+1) * 2)
    # train the generator and classifier
   
    generator, discriminator = train_replayer(
        TrainDataLoaders, len(task_classes_arr[task_id]), writer)
    print("generate image from 0 to ", task_classes_arr[task_id])
    for img_id in task_classes_arr[task_id]:
        img, _ = sample_image(generator, 10, [img_id], n_latent)   
        save_image(img.data, 'store/x%d-%d.png' %
                   (task_id, img_id), nrow=10, normalize=True)
            
