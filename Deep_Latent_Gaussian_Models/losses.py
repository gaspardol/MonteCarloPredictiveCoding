
import torch
from torch import nn
import torch.nn.functional as F


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu_list, R_list):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    '''
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    C = R @ R.transpose(-1,-2) # batch_size x size x size
    #KLD = -0.5 * torch.sum(1 + C.diagonal(dim1=-2,dim2=-1).sum(-1) - mu.pow(2).sum(-1) - 2*R.diagonal(dim1=-2,dim2=-1).log().sum(-1))
    KLD = 0.5 * torch.sum(mu.pow(2).sum(-1) + C.diagonal(dim1=-2,dim2=-1).sum(-1)  - 2*R.diagonal(dim1=-2,dim2=-1).log().sum(-1) -1)
    '''
    if not isinstance(mu_list, (list, tuple)):
        mu_list = [mu_list]
        R_list = [R_list]
    
    KLD_list = []
    for mu,R in zip(mu_list, R_list):
        C = R @ R.transpose(-1,-2) # batch_size x size x size
        KLD = 0.5 * torch.sum(mu.pow(2).sum(-1) + C.diagonal(dim1=-2,dim2=-1).sum(-1)  - 2*R.diagonal(dim1=-2,dim2=-1).log().sum(-1) -1)
        KLD_list.append(KLD)

    return BCE + sum(KLD_list)

