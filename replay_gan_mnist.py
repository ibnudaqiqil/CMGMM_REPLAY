from torch.utils.tensorboard import SummaryWriter
from datasets.MNIST import MNIST_IncrementalDataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import helper.LIBS as lib
import models.GAN as models
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import os
import sys



import torch
import torchvision
import numpy as np

writer = SummaryWriter()
batch_size = num_noise = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

TrainDataLoaders = []
TestDataLoaders = []

for i in range(5):
    # MNIST dataset
    TrainDataSet = MNIST_IncrementalDataset(source='./data/',
                                                     train=True,
                                                     transform=transform,
                                                     download=True,
                                                     classes=range(i * 2, (i+1) * 2))

    TestDataSet = MNIST_IncrementalDataset(source='./data/',
                                                    train=False,
                                                    transform=transform,
                                                    download=True,
                                                    classes=range(i * 2, (i+1) * 2))

    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet,
                                                        batch_size=batch_size,
                                                        shuffle=True))
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet,
                                                       batch_size=batch_size,
                                                       shuffle=False))

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
ld = 10
epochs = 100
gan_p_real_list = []
gan_p_fake_list = []
solver_acc_dict = {}

pre_gen = None
pre_solver = None

#iteration every task in MNIST
# Task 0 (0,1), Task 1 (2,3), Task 2 (4,5), Task 3 (6,7), Task 4 (8,9)
for t in range(5):
    ratio = 1 / (t+1)  # current task's ratio
    if t > 0:
        pre_gen = generator
        pre_solver = solver

        lib.model_grad_switch(pre_gen, False)
        lib.model_grad_switch(pre_solver, False)
    
    solver = models.Solver(t+1)

    if torch.cuda.is_available():

        solver = solver.to(device)

    lib.init_params(solver)
    TrainDataLoader = TrainDataLoaders[t]
    generator,discriminator = models.train_generator(
        TrainDataLoader, ratio, num_noise, pre_gen, writer)
    optim_s = torch.optim.Adam(solver.parameters(), lr=0.001)

         
    print("TRAINING Classifier =============")

    for image, label in TrainDataLoader:
        celoss = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            image = image.to(device)
            label = label.to(device)
            celoss = celoss.to(device)

        solver.zero_grad()
        optim_s.zero_grad()

        output = solver(image)
        loss = celoss(output, label) * ratio
        loss.backward()
        optim_s.step()

        if pre_solver is not None:
            solver.zero_grad()
            optim_s.zero_grad()

            noise = lib.sample_noise(batch_size, num_noise, device)
            g_image = pre_gen(noise)
            g_label = pre_solver(g_image).max(dim=1)[1]
            g_output = solver(g_image)
            loss = celoss(g_output, g_label) * (1 - ratio)

            loss.backward()
            optim_s.step()

    ### Evaluate solver
    solver_acc_dict[t+1] = lib.solver_evaluate(t, gen, solver, ratio, device, TestDataLoaders)
    writer.add_scalar('Accuracy/test', solver_acc_dict[t+1], t)



