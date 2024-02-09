
import torch
from torch import nn
import torch.nn.functional as F

from cholesky_factor import CholeskyFactor, DiagonalFactor

class RecognitionModel(nn.Module):
    def __init__(self, latent_dim = 20, hidden_dim = 400, chol_factor_cls = None):
        super().__init__()

        self.chol_factor = chol_factor_cls(latent_dim)

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, self.chol_factor.free_parameter_size())

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)

        logvar_free = self.fc22(h1)
        R = self.chol_factor.parameterize(logvar_free)
        return mu, R
    
    def sample(self, mu, R):
        eps = torch.randn_like(mu)
        return mu + torch.einsum('ijk,ik->ij', R, eps) 
    
    def log_prob(self, z, mu, R):
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=R)
        return dist.log_prob(z)

class RecognitionModelStacked(nn.Module):
    def __init__(self, latent_dim_list, hidden_dim_list, chol_factor_cls = None):
        super().__init__()

        self.node_list = nn.ModuleList()
        for latent_dim, hidden_dim in zip(latent_dim_list, hidden_dim_list):
            node = RecognitionModel(latent_dim = latent_dim, hidden_dim = hidden_dim, 
                chol_factor_cls = chol_factor_cls)
            self.node_list.append(node)

    def forward(self, x):
        mu_list= []
        R_list = []
        for node in self.node_list:
            mu, R = node(x)
            mu_list.append(mu)
            R_list.append(R)
        return mu_list, R_list

    def sample(self, mu_list, R_list):
        z_list = []
        for node, mu, R in zip(self.node_list, mu_list, R_list):
            z = node.sample(mu, R)
            z_list.append(z)
        return z_list

    def log_prob(self, z_list, mu_list, R_list):
        log_prob = 0.0
        for z,mu,R in zip(z_list, mu_list, R_list):
            dist = torch.distributions.MultivariateNormal(mu, scale_tril=R)
            log_prob += dist.log_prob(z)#.sum()
        return log_prob

class RecognitionMNIST(RecognitionModelStacked):
    def __init__(self, chol_factor_cls = None):
        #latent_dim_list = [200, 201]
        latent_dim_list = [201, 200]
        hidden_dim_list = [400, 400]
        super().__init__(latent_dim_list, hidden_dim_list, chol_factor_cls = chol_factor_cls)

class RecognitionMNISTVAE(RecognitionModelStacked):
    def __init__(self, chol_factor_cls = None):
        #latent_dim_list = [200, 201]
        latent_dim_list = [20]
        hidden_dim_list = [400]
        super().__init__(latent_dim_list, hidden_dim_list, chol_factor_cls = chol_factor_cls)

class RecognitionMNISTVAELarge(RecognitionModelStacked):
    def __init__(self, chol_factor_cls = None):
        #latent_dim_list = [200, 201]
        latent_dim_list = [200]
        hidden_dim_list = [1000]
        super().__init__(latent_dim_list, hidden_dim_list, chol_factor_cls = chol_factor_cls)

'''
class RecognitionNode(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim = 400, chol_factor_cls = None):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        #self.chol_factor_cls = chol_factor_cls

        self.chol_factor = chol_factor_cls(latent_dim)
'''