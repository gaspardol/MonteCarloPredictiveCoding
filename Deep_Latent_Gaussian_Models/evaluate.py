
import torch
from torch import nn
from torch.nn import functional as F

from torchvision import datasets, transforms

import math

import argparse

import generative
import recognition
import cholesky_factor

from tqdm import tqdm

parser = argparse.ArgumentParser(description='DLGM MNIST Evaluate')
parser.add_argument('generative_model')
parser.add_argument('recognition_model')
parser.add_argument('chol_factor_cls')
parser.add_argument('checkpoint')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--particle-size', type=int, default=16)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                    help='learning rate')


args = parser.parse_args()

assert args.batch_size % args.particle_size == 0

batch_size_load = args.batch_size // args.particle_size

args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if args.cuda else {}
train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                   transform=transforms.ToTensor())
test_dataset = datasets.MNIST('../dataset', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=batch_size_load, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=batch_size_load, shuffle=True, **kwargs)

chol_factor_cls = cholesky_factor.__dict__[args.chol_factor_cls]
generative_model = generative.__dict__[args.generative_model]().to(device)
recognition_model = recognition.__dict__[args.recognition_model](chol_factor_cls = chol_factor_cls).to(device)

checkpoint = torch.load(args.checkpoint)
generative_model.load_state_dict(checkpoint['generative_model'])
recognition_model.load_state_dict(checkpoint['recognition_model'])

generative_model.eval()
recognition_model.eval()

@torch.no_grad()
def evaluate(loader):
    '''
    estimate log likelihood for model
    '''
    log_prob = 0.0
    p_v_given_z_sum = 0.0

    for data, _ in tqdm(loader):

        data = data.repeat(args.particle_size, 1, 1, 1).to(device)
        mu, R = recognition_model(data.view(-1, 28*28)) # or mu_list, R_list
        z = recognition_model.sample(mu, R) # or z_list
        recon_batch = generative_model(z)

        p_v_given_z = -F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='none').sum(-1)
        q_z_given_v = recognition_model.log_prob(z, mu, R)
        p_z = generative_model.log_prob_prior(z)

        load_size = data.shape[0]//args.particle_size
        #print('load_size:', load_size, data.shape[0], args.particle_size, data.shape[0]//args.particle_size)

        logf = (p_v_given_z + p_z - q_z_given_v)
        log_prob += logf.view(args.particle_size, load_size).logsumexp(0).sum()
        log_prob -= math.log(args.particle_size) * load_size

        p_v_given_z_sum += p_v_given_z.view(args.particle_size, load_size).logsumexp(0).sum()
        p_v_given_z_sum -= math.log(args.particle_size) * load_size

        #print([fold.view(args.particle_size, load_size).logsumexp(0).sum() for fold in [p_v_given_z, p_z, q_z_given_v]])
        #print(logf.view(args.particle_size, load_size).logsumexp(0).sum(), math.log(args.particle_size) * load_size)

        #log_prob_batch = (p_v_given_z + p_z - q_z_given_v) / args.particle_size

        #log_prob += log_prob_batch

    return {
        '-ln p(v)': -log_prob / len(loader.dataset), 
        '-ln p(v|x)': -p_v_given_z_sum / len(loader.dataset)
    }

print("Train dataset: -ln p(v):", evaluate(train_loader))
print("Test dataset: -ln p(v):", evaluate(test_loader))

'''
python evaluate.py GenerativeModel RecognitionModel CholeskyFactor checkpoints/GenerativeModel_RecognitionModel_CholeskyFactor.pth
python evaluate.py GenerativeMNIST RecognitionMNIST CholeskyFactor checkpoints/GenerativeMNIST_RecognitionMNIST_CholeskyFactor.pth
python evaluate.py GenerativeMNISTLarge RecognitionMNIST CholeskyFactor checkpoints/GenerativeMNISTLarge_RecognitionMNIST_CholeskyFactor.pth
python evaluate.py GenerativeMNISTVAE RecognitionMNISTVAE DiagonalFactor checkpoints/GenerativeMNISTVAE_RecognitionMNISTVAE_DiagonalFactor.pth
'''