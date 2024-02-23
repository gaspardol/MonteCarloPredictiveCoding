import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random, os


import predictive_coding as pc

from utils.plotting import setup_fig
from utils.model import sample_x_fn_normal, fe_fn, random_step
from utils.training_evaluation import get_mcpc_trainer_one_sample


random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


def varying_langevin_noise(path_figures, noise_vars):
    # network parameters
    hidden_size = 1
    output_size = 1
    # data parameters
    batch_size = 2048
    n = 25 # number of batches
    #training
    epochs = 10
    start = [-7,-5] # initial parameters for network 

    # sample a batch of data
    mu=1.
    var=5.
    datas = [mu + np.sqrt(var)*torch.randn(batch_size, output_size) for i in range(n)]
    pseudo_input = torch.zeros(batch_size, hidden_size)
    
    results_var=[]
    results_weights=[]
    # noise variance
    for idx, noise_var in enumerate(iter(noise_vars)):
        # initialise model
        gen_mcpc = nn.Sequential(
            nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
            pc.PCLayer(sample_x_fn=sample_x_fn_normal),
            nn.Linear(hidden_size, output_size,bias=False),
        )
        # initialise model
        gen_mcpc.train()
        nn.init.constant_(gen_mcpc[0].bias, start[0])
        nn.init.constant_(gen_mcpc[2].weight, start[1]) 

        config_mcpc = {
            "input_var":1.,
            "K": 150,
            "optimizer_x_kwargs_mcpc":{"lr": np.clip(0.01*noise_var/2, 0.001, 0.05)}, #001 for 0.5
            "optimizer_p_fn_mcpc": optim.Adam,
            "optimizer_p_kwargs_mcpc": {"lr": np.clip(0.3/noise_var, 1/2, 3)}, #, "momentum":0.0
            "loss_fn": fe_fn 
        }
        
        # create MCPC trainer
        mcpc_trainer = get_mcpc_trainer_one_sample(gen_mcpc, config_mcpc, training=True)
        
        # training
        for e in range(epochs):
            for data in tqdm(datas): #
                mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config_mcpc["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config_mcpc["input_var"]},
                                            callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer, 'var':noise_var}, #
                                            is_sample_x_at_batch_start=True,is_log_progress=False,is_return_results_every_t=False,is_checking_after_callback_after_t=False)

        ## generate samples of sensory stimuli
        # initialise sampling
        config_generation = copy.deepcopy(config_mcpc)
        config_generation["K"] = 10000
        config_generation["optimizer_x_kwargs_mcpc"]["lr"] = max([0.01, config_generation["optimizer_x_kwargs_mcpc"]["lr"]])

        def gen_pc_layer_loss(inputs):
            return (1/config_mcpc["input_var"])*0.5 *(inputs['mu'] - inputs['x'])**2 

        gen_pc = gen_mcpc
        gen_pc.add_module("output", pc.PCLayer(energy_fn=gen_pc_layer_loss, sample_x_fn=sample_x_fn_normal))
        gen_pc.train()

        # create MCPC trainer
        print("making trainers")
        mcpc_trainer_ = get_mcpc_trainer_one_sample(gen_pc, config_generation, training=False)
        
        # mc inference
        print("Inference")
        mc_results = mcpc_trainer_.train_on_batch(inputs=pseudo_input,callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer_, 'var':noise_var},is_sample_x_at_batch_start=True,is_return_results_every_t=True,is_return_outputs =True)
        
        # samples
        generated_dist = mc_results["outputs"][-1].squeeze().cpu().detach().numpy()
        
        # extract std and mean of samples to compare 
        results_var.append(np.var(generated_dist))

        # extract weights learned by model
        results_weights.append([gen_pc[0].bias.item(), gen_pc[2].weight.item()])

        # true distribution
        y = np.linspace(-10,10,500)
        data_pdf = (1/np.sqrt(2*np.pi*var))*np.exp(-0.5*(y-mu)**2/(var))

        # compare samples to true dist
        if len(noise_vars)<=4:
            setup_fig(zero=True)
            plt.figure()
            plt.plot(y, data_pdf, 'k',label=r"$p(y)$",linewidth=3)
            plt.hist(generated_dist, bins=np.linspace(-12, 12, 21), density=True, label="MCPC")
            plt.legend()
            plt.xlabel("$x_0$, y")
            plt.ylabel("probability " + r"$p(x_0;\theta)$")
            plt.yticks([0, 0.05, 0.1, 0.15])
            plt.xlim([-12,12])
            plt.ylim([0,0.196])
            plt.title(r"$\sigma^2$ : " + str(noise_var/2))
            plt.tight_layout()
            plt.savefig(path_figures +"//SIa_"+str(idx) + ".svg")
            plt.show(block=False)

    if len(noise_vars)>4:
        setup_fig(zero=True)
        plt.figure()
        plt.plot(noise_vars,results_var,linewidth=3)
        plt.hlines(np.sqrt(var)**2, min(noise_vars), max(noise_vars), colors="black", label="data")
        plt.vlines(2*var, min(results_var), max(results_var), colors="grey", linestyles="dashed", label="learning limit")
        plt.legend()
        plt.xlabel("Langevin noise variance $2\sigma^2$")
        plt.ylabel(r"variance of x$_0$, y")
        plt.xscale('log')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(path_figures +"//SIb.svg")
        plt.show(block=False)

        setup_fig(zero=True)
        plt.figure()
        # plt.plot(noise_vars, np.array(results_weights)[:,0], label="bias",linewidth=3)
        plt.plot(noise_vars, np.abs(np.array(results_weights)[:,1]),linewidth=3)
        plt.vlines(2*var, min(np.abs(np.array(results_weights)[:,1])), max(np.abs(np.array(results_weights)[:,1])), colors="grey", linestyles="dashed", label="learning limit")
        plt.xlabel("Langevin noise variance $2\sigma^2$")
        plt.ylabel(r"learned |$W_0$|")
        plt.xscale('log')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(path_figures +"//SIc.svg")
        plt.show(block=False)
    
    plt.show()
    
    

if __name__ == "__main__":
    pwd = os.getcwd()
    path_figures = pwd + "//figures"
    
    varying_langevin_noise(path_figures, np.logspace(-1, 1.5, 40))
    varying_langevin_noise(path_figures, [0.2, 2, 8, 20])