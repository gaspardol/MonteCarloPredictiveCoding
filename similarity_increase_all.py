import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from torch.utils.data import  DataLoader
from torchvision.utils import save_image
import random, os

import predictive_coding as pc
from ResNet9 import ResNet9

from utils.training_evaluation import kl_divergence_discrete, MNIST_LinearClassifier
from utils.model import sample_x_fn_cte, sample_x_fn_normal, sample_x_fn, bernoulli_fn, bernoulli_fn_mask, fe_fn,fe_fn_mask, get_model, get_representations, random_step
from utils.training_evaluation import train, test, get_pc_trainer, get_mcpc_trainer, sample_pc
from utils.data import get_mnist_data
from utils.plotting import setup_fig, proba_to_coordinate

from matplotlib.ticker import StrMethodFormatter

random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


pwd = os.getcwd()
path_models = pwd + "//models"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


import copy
from utils.data import GratingDataset, NoiseDataset
from utils.model import zero_fn
from utils.training_evaluation import KLdivergence

epochs=[0, 15]

config = {
    #
    "EPOCHS":50,
    "batch_size_train":256,
    "batch_size_val": 1024,
    "batch_size_test": 256,
    #
    "input_size": 20,
    "hidden_size": 128,
    "hidden2_size": 128,
    "output_size": 784,
    "loss_fn": bernoulli_fn ,
    "activation_fn": 'relu',
    #
    "T_pc":1,
    "optimizer_x_fn_pc": optim.Adam,
    "optimizer_x_kwargs_pc":{"lr": 0.7},
    #
    "mixing":50,
    "sampling":100,
    "optimizer_x_kwargs_mcpc":{"lr": 0.1},
    #
    "optimizer_p_fn_mcpc": optim.Adam,
    "optimizer_p_kwargs_mcpc": {"lr": 0.01, "weight_decay":0.},
    "input_var":None
}

# Load MNIST data
train_loader, val_loader, test_loader = get_mnist_data(config)
# make gratings
grating_dataset = GratingDataset(config["batch_size_test"], size=28, num_orientations=16)
grating_loader = DataLoader(grating_dataset, batch_size=config["batch_size_test"], shuffle=True)
# make noise
noise_dataset = NoiseDataset(config["batch_size_test"], size=28)
noise_loader = DataLoader(noise_dataset, batch_size=config["batch_size_test"], shuffle=True)

# select 5 neurons randomly
rand_idx = [None for i in range(3)]
rand_idx[-1] = random.sample(range(config["hidden2_size"]), 5)
rand_idx[-2] = random.sample(range(config["hidden_size"]), 5)
rand_idx[-3] = random.sample(range(config["input_size"]), 5)

kls_seed=[]
seeds = range(10)
for seed in seeds:
    model_name_base= path_models + "\\epoch_save\\mcpc_aging_"+str(seed)+"_"

    indent = 20 # indent to keep only part of data to prevent memory overload
    kls = np.zeros((3, 3,len(epochs)))
    for idx, epoch in enumerate(epochs):
        # get model
        gen_pc = get_model(config, use_cuda)
        gen_pc.train()

        # create trainers for genaration
        config_gen = copy.deepcopy(config)
        config_gen["mixing"] = 500
        config_gen["sampling"] = 9500
        config_gen["optimizer_x_kwargs_mcpc"] = {"lr": 0.05}

        pc_trainer_gen = get_pc_trainer(gen_pc, config_gen, is_mcpc=True, training=False)
        mcpc_trainer_gen = get_mcpc_trainer(gen_pc, config_gen, training=False)
        
        model_name = model_name_base + "epoch"+str(epoch) if epoch!=0 else model_name_base + "epoch_init"
        gen_pc.load_state_dict(torch.load(model_name), strict=False)

        # load all data
        data, _ = list(test_loader)[0]
        grating_data = list(grating_loader)[0]
        noise_data = list(noise_loader)[0]
        pseudo_input = torch.zeros(data.shape[0], config_gen["input_size"])
        if use_cuda:
            pseudo_input, data, grating_data, noise_data = pseudo_input.cuda(), data.cuda(), grating_data.cuda(), noise_data.cuda()

        # get spontaneous activity
        pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=zero_fn,loss_fn_kwargs={})
        res_mcpc_prior = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=zero_fn, loss_fn_kwargs={},callback_after_t=random_step,
                                    callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                    is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)

        ## get evoked activity
        # for natural stimuli
        pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        res_mcpc = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': data}, 
                                    callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                    is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
        # for gratings
        pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': grating_data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        res_mcpc_grating = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': grating_data}, 
                                    callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                    is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)
        # for noise
        pc_trainer_gen.train_on_batch(inputs=pseudo_input, loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': noise_data},is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)
        res_mcpc_noise = mcpc_trainer_gen.train_on_batch(inputs=pseudo_input,loss_fn=config_gen["loss_fn"],loss_fn_kwargs={'_target': noise_data}, 
                                    callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_gen}, #
                                    is_sample_x_at_batch_start=False, is_log_progress=True, is_return_results_every_t=True, is_return_xs=True)

        for latent in range(3):
            prior = torch.concatenate([r[latent] for r in res_mcpc_prior["xs"][config_gen["mixing"]::]])
            posterior_natural = torch.concatenate([r[latent] for r in res_mcpc["xs"][config_gen["mixing"]::]])
            posterior_gratings = torch.concatenate([r[latent] for r in res_mcpc_grating["xs"][config_gen["mixing"]::]])
            posterior_noise = torch.concatenate([r[latent] for r in res_mcpc_noise["xs"][config_gen["mixing"]::]])
            
            ## data proprocessing
            prior = prior[:,rand_idx[latent]]
            posterior_natural = posterior_natural[:,rand_idx[latent]]
            posterior_gratings = posterior_gratings[:, rand_idx[latent]]
            posterior_noise = posterior_noise[:, rand_idx[latent]]

            prior_unit = (prior - prior.mean(0))/ (prior.std(0) + 1e-6)
            posterior_natural_unit = (posterior_natural - prior.mean(0))/ (prior.std(0) + 1e-6)
            posterior_gratings_unit = (posterior_gratings - prior.mean(0))/ (prior.std(0) + 1e-6)
            posterior_noise_unit = (posterior_noise - prior.mean(0))/ (prior.std(0) + 1e-6)

            kls[latent, 0,idx] = KLdivergence(prior_unit[::indent], posterior_natural_unit[::indent])
            kls[latent, 1,idx] = KLdivergence(prior_unit[::indent], posterior_noise_unit[::indent])
            kls[latent, 2,idx] = KLdivergence(prior_unit[::indent], posterior_gratings_unit[::indent])

            if kls[latent,0,idx] < 0 or kls[latent,1,idx] < 0 or kls[latent,2,idx] < 0:
                print("negative KL divergence")
                print(kls[latent, 0,idx], kls[latent, 1,idx], kls[latent, 2,idx])
                raise ValueError

        del prior, posterior_noise, posterior_gratings, posterior_natural
    kls_seed.append(kls)


kls_seed = np.array(kls_seed)

for latent in range(3):
    setup_fig()
    plt.figure()
    # concatenate of data across seeds
    kls_np = np.concatenate([k.reshape(k.shape[0],k.shape[1], 1) for k in kls_seed[:,latent,:,:]], axis=2)

    # find mean and s.e.m.
    kls_mean = kls_np.mean(-1)
    kls_sem =  kls_np.std(-1)/kls_np.shape[-1]

    # Set up the bar chart
    conditions = [str(i) for i in epochs]
    types = ['natural ', 'noise', 'gratings']
    colors = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots()
    bar_width = 0.2
    index = np.arange(len(conditions))
    for i, type_label in enumerate(types):
        ax.bar(index + i * bar_width, kls_mean[i, :], bar_width, label=type_label, color=colors[i], yerr=[np.zeros_like(kls_sem[i, :])+0.05, kls_sem[i, :]], zorder=2, error_kw=dict(capsize = 4,zorder=1))

    ax.set_ylim(0, None)
    ax.set_ylabel('KL divergence')
    ax.set_xlabel('Epochs')
    ax.set_xticks(index + (bar_width * (len(types) - 1)) / 2)
    ax.set_xticklabels(conditions)  
    plt.tight_layout()
    plt.savefig("figures/SI_kl_divergence_latent_"+ str(latent) +".svg")

# save data
np.save('similarity_all_layers.npy', kls_seed)
