
import numpy as np


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#C, H, W = args.channels, args.img_size, args.img_size

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)


class Generator(nn.Module):
    # initializers
    def __init__(self,H=28,W=28):
        super(Generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, H*W)

    # forward method

    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, H=28, W=28):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(H*W, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # forward method

    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input.view(input.size(0), -1)), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x


def train_replayer(dataloader, n_classes,writer):
    learning_rate = 0.0002
    beta1 = 0.5  # decay of first order momentum of gradient'
    beta2 = 0.999  # decay of second order momentum of gradient
    n_epochs =100
    batch_size = 64
    sample_interval= 10
    latent_dim = 100    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize Generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(beta1,beta2))


    batches_done = 0
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            Batch_Size = batch_size
            N_Class = n_classes
            # Adversarial ground truths
            valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

            real_y = torch.zeros(Batch_Size, N_Class)
            real_y = Variable(real_y.scatter_(
                1, labels.view(Batch_Size, 1), 1).cuda())
            #y = Variable(y.cuda())

            # Sample noise and labels as generator input
            noise = Variable(torch.randn((Batch_Size, latent_dim)).cuda())
            gen_labels = (torch.rand(Batch_Size, 1) *
                        N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(Batch_Size, N_Class)
            gen_y = Variable(gen_y.scatter_(
                1, gen_labels.view(Batch_Size, 1), 1).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real images
            d_real_loss = adversarial_loss(
                discriminator(real_imgs, real_y).squeeze(), valid)
            # Loss for fake images
            gen_imgs = generator(noise, gen_y)
            d_fake_loss = adversarial_loss(discriminator(
                gen_imgs.detach(), gen_y).squeeze(), fake)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            #gen_imgs = generator(noise, gen_y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(
                gen_imgs, gen_y).squeeze(), valid)

            g_loss.backward()
            optimizer_G.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                                            d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                noise = Variable(torch.FloatTensor(np.random.normal(
                    0, 1, (N_Class**2, latent_dim))).cuda())
                #fixed labels
                y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class, 1).expand(-1, N_Class).contiguous()
                y_fixed = torch.zeros(N_Class**2, N_Class)
                y_fixed = Variable(y_fixed.scatter_( 1, y_.view(N_Class**2, 1), 1).cuda())

                #gen_imgs = generator(noise, y_fixed).view(-1, C, H, W)
                gen_imgs = generator(noise, y_fixed).view(-1, 1, 28, 28)

                #save_image(gen_imgs.data, img_save_path + '/%d-%d.png' % (epoch, batches_done), nrow=N_Class, normalize=True)
    return generator, discriminator


def sample_image(generator,n_row, gen_label, latent_dim):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
   
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(
        0, 1, (n_row*len(gen_label), latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
  
    y = Variable(LongTensor(
        np.array([lbl for lbl in gen_label for _ in range(n_row)])))
    gen_imgs = generator(z, y)
        #save_image(gen_imgs.data, img_save_path + '/%d.png' % lbl, nrow=n_row, normalize=True)
    gen_imgs = generator(z, y)
    return gen_imgs