import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

class increamentalMixtureNetwork(nn.Module):
    def __init__(self, n_components=1, input_size=1, n_classes=10,
                 
                 fullcovs=False, device=None):
        super(increamentalMixtureNetwork, self).__init__()
        self.n_components = n_components
        self.pi_network = CategoricalNetwork(input_size, n_components)
        self.normal_network = MixtureDiagNormalNetwork( input_size, n_classes, n_components)

        self.input_size = input_size
        self.n_classes = n_classes        
        self.fullcovs = fullcovs
        self.device = device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


    def forward(self,x):
        #x = x.view(x.size(0), self.input_size)
        #x = self.features(x)   
        return self.mixture_network(x)

    def mixture_network(self, x):
        #x = x.view(x.size(0), self.input_size)
        #x = self.features(x)
        #x = x.view(x.size(0), self.n_components, self.input_size)
        x = x.view(x.size(0), self.input_size)
        x = self.features(x)
        x = x.view(x.size(0), self.n_components, self.hidden_size)
        x = self.mixture_weights(x)
        x = self.mixture_means(x)
        x = self.mixture_vars(x)
        return x


class MixtureDensityNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out, n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))


class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)


