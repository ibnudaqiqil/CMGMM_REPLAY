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
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


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

    disc = models.Discriminator_Conv(input_shape=(1, 28, 28))
    print("Create solver for Task %d" % (t+1))
    solver = models.Solver(t+1)

    if torch.cuda.is_available():
        solver = solver.to(device)

    lib.init_params(solver)

    TrainDataLoader = TrainDataLoaders[t]

    optim_solver = torch.optim.Adam(solver.parameters(), lr=0.001)

    # Generator Training
 
    example_num = 2000
    print("TRAINING GENERATOR =============")
    for i in range((t+1)*2):
        f = open("./store/gmm2/model_"+str(i)+".pickle", "rb")
        dir_name = "Task "+str(i)
        generator = pickle.load(f)
        f.close()
        digits, _ = generator.sample(48)
        np.clip(digits, 0, 1, out=digits)
        #print(digits_a.shape, digits_b.shape)
        digits = digits.astype(np.float32)
        digits[np.abs(digits) < 0.1] = 0.0
        #digits[np.abs(digits) < 0.2] = 0.4
        #digits[np.abs(digits) < 0.3] = 0.5
        digits[np.abs(digits) > 0.7] = 1.0


        digits = digits.reshape(digits.shape[0], 1, 28, 28)
        t = torch.from_numpy(digits)
        grid = torchvision.utils.make_grid(t)
        plt.imshow(grid.permute(1, 2, 0),cmap='gray')
        plt.show()
        writer.add_image(dir_name, grid, 0)
'''


    # train solver
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
    solver_acc_dict[t+1] = lib.solver_evaluate(
        t, gen, solver, ratio, device, TestDataLoaders)
    writer.add_scalar('Accuracy/test', solver_acc_dict[t+1], t)
'''
