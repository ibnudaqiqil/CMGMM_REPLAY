import torch
import torchvision
import numpy as np


class Generator_FC(torch.nn.Module):
    """
    Fully-Connected Generator
    """

    def __init__(self, input_node_size, output_shape, hidden_node_size=256,
                 hidden_node_num=3, normalize=True):
        """
        input_node_size: shape of latent vector
        output_shape: shape of output data, (# of channels, width, height)
        """
        super(Generator_FC, self).__init__()
        self.output_shape = output_shape
        output_node_size = output_shape[0] * output_shape[1] * output_shape[2]

        HiddenLayerModule = []
        for _ in range(hidden_node_num):
            HiddenLayerModule.append(torch.nn.Linear(
                hidden_node_size, hidden_node_size))
            if normalize:
                HiddenLayerModule.append(
                    torch.nn.BatchNorm1d(hidden_node_size, 0.8))
            HiddenLayerModule.append(torch.nn.LeakyReLU(0.2))

        self.network = torch.nn.Sequential(

            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2),

            *HiddenLayerModule,

            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Tanh()

        )

    def forward(self, x):
        num_data = x.shape[0]
        _x = x.view(num_data, -1)
        return self.network(_x)


class Generator_Conv(torch.nn.Module):
    """
    Generator Class for GAN
    """

    def __init__(self, input_node_size, output_shape, hidden_node_size=256,
                 hidden_node_num=3):
        """
        input_node_size: dimension of latent vector
        output_shape: dimension of output image
        """
        super(Generator_Conv, self).__init__()

        self.input_node_size = input_node_size
        self.output_shape = output_shape
        num_channels, width, _ = output_shape

        layer_channels = []
        if width <= 32:
            layer_channels.append(width//2)
            layer_channels.append(width//4)

        # HiddenLayerModule = []
        # for _ in range(hidden_node_num):
        #     HiddenLayerModule.append(torch.nn.ConvTranspose2d(

        #                             ))
        #     if normalize:
        #         HiddenLayerModule.append(torch.nn.BatchNorm2d(num_features=))

        #     HiddenLayerModule.append(torch.nn.ReLU())

        # self.network = torch.nn.Sequential(
        #     torch.nn.ConvTranspose2d(input_node_size, out_channels=, kernel_size=, stride=, padding=, bias=False),
        #     torch.nn.LeakyReLU(0.2),

        #     *HiddenLayerModule,

        #     torch.nn.ConvTranspose2d(input_node_size, out_channels=, kernel_size=, stride=, padding=, bias=False),
        #     torch.nn.Tanh(),
        # )

        conv2d_1 = torch.nn.ConvTranspose2d(in_channels=input_node_size,
                                            out_channels=width*8,
                                            kernel_size=layer_channels[1],
                                            stride=1,
                                            padding=0,
                                            bias=False)
        conv2d_2 = torch.nn.ConvTranspose2d(in_channels=width*8,
                                            out_channels=width*4,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            bias=False)
        conv2d_3 = torch.nn.ConvTranspose2d(in_channels=width*4,
                                            out_channels=num_channels,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=width*8),
            torch.nn.ReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=width*4),
            torch.nn.ReLU(inplace=True),
            conv2d_3,
            torch.nn.Tanh()
        )

    def forward(self, x):
        _x = x.view(-1, self.input_node_size, 1, 1)
        return self.network(_x)


class Discriminator_FC(torch.nn.Module):
    """
    Fully-Connected Discriminator
    """

    def __init__(self, input_shape, hidden_node_size=256, output_node_size=1):
        super(Discriminator_FC, self).__init__()
        input_node_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.network(_x).view(-1, 1)


class Discriminator_Conv(torch.nn.Module):
    """
    Discriminator Class for GAN
    """

    def __init__(self, input_shape, hidden_node_size=256, output_node_size=1):
        """
        Parameters
        ----------
        input_shape: (C, W, H)

        """
        super(Discriminator_Conv, self).__init__()
        num_channels, width, _ = input_shape

        conv2d_1 = torch.nn.Conv2d(in_channels=num_channels,
                                   out_channels=width*4,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=width*4,
                                   out_channels=width*8,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=width*8,
                                   out_channels=output_node_size,
                                   kernel_size=7,
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=width*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=width*8),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)


class Solver(torch.nn.Module):
    """
    Solver Class for Deep Generative Replay
    """

    def __init__(self, T_n):
        super(Solver, self).__init__()
        fc1 = torch.nn.Linear(28*28, 100)
        fc2 = torch.nn.Linear(100, 100)
        fc3 = torch.nn.Linear(100, T_n * 2)
        self.network = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3
        )

    def forward(self, x):
        return self.network(x.view(x.shape[0], -1))


def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.uniform_(p, 0.1, 0.2)


def sample_noise(batch_size, N_noise, device='cpu'):
    """
    Returns 
    """
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda:0'

    return torch.randn(batch_size, N_noise).to(device)


def train_generator(TrainDataLoader, ratio, num_noise, prev_generator, writer):
    #num_noise =48
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ld = 10
    epochs = 10
    learning_rate = 0.001
    generator = Generator_Conv(input_node_size=num_noise, output_shape=(
        1, 28, 28), hidden_node_size=256, hidden_node_num=2)
    discriminator =Discriminator_Conv(input_shape=(1, 28, 28))

    if torch.cuda.is_available():
        generator = generator.to(device)
        discriminator = discriminator.to(device)
     
    init_params(generator)
    init_params(discriminator)

    optim_g = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0, 0.9))
    optim_descriminator = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0, 0.9))
    
    # Generator Training
    print("TRAINING GENERATOR =============")
    for epoch in range(epochs):
        #set in to training mode
        generator.train()
        discriminator.train()

        for i, (x, _) in enumerate(TrainDataLoader):
            num_data = x.shape[0]
            noise = sample_noise(num_data, num_noise, device)

            if torch.cuda.is_available():
                x = x.to(device)
                noise = noise.to(device)

            if prev_generator is not None:
                with torch.no_grad():
                    # append generated image & label from previous scholar
                    datapart = int(num_data*ratio)
                    perm = torch.randperm(num_data)[:datapart]
                    x = x[perm]

                    x_g = prev_generator(sample_noise(num_data, num_noise, device))
                    perm = torch.randperm(num_data)[:num_data - datapart]
                    x_g = x_g[perm]

                    x = torch.cat((x, x_g))

            ### Discriminator train
            optim_descriminator.zero_grad()
            discriminator.zero_grad()
            x_g = generator(noise)

            ## Regularization term
            eps = torch.rand(1).item()
            x_hat = x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)
            x_hat.requires_grad = True

            loss_xhat = discriminator(x_hat)
            fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
            if torch.cuda.is_available():
                fake = fake.to(device)

            gradients = torch.autograd.grad(outputs=loss_xhat,
                                            inputs=x_hat,
                                            grad_outputs=fake,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]
            gradients = gradients.view(gradients.shape[0], -1)
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld

            p_real = discriminator(x)
            p_fake = discriminator(x_g.detach())

            loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
            loss_d.backward()
            optim_descriminator.step()

            ### Generator Training
            if i % 5 == 4:
                discriminator.zero_grad()
                optim_g.zero_grad()
                p_fake = discriminator(x_g)

                loss_g = -torch.mean(p_fake)
                loss_g.backward()
                optim_g.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, epochs, loss_d.item(), loss_g.item()))
        writer.add_scalar('D Loss/train', loss_d.item(), epoch)
        writer.add_scalar('G Loss/train', loss_g.item(), epoch)
        if epoch % 10 == 0:
            dir_name = "Task_1"+str(epoch)

            noise = sample_noise(64, num_noise, device)
            gen_image = generator(noise)
            grid = torchvision.utils.make_grid(gen_image)
            writer.add_image(dir_name, grid, 0)
            writer.add_graph(generator, gen_image)
            #torchvision.utils.save_image(gen_image, 'imgs/Task_%d/%03d.png' % (t+1, epoch+1))
    return generator,discriminator
