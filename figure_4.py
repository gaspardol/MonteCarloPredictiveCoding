import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random, os

import predictive_coding as pc
from Deep_Latent_Gaussian_Models.DLGM import DLGM
from utils.plotting import setup_fig
from utils.model import random_step, get_model, bernoulli_fn, bernoulli_fn_mask, fe_fn_mask, sample_x_fn_normal, random_step
from utils.training_evaluation import sample_pc, get_mcpc_trainer, get_pc_trainer, fe_fn
from utils.data import get_mnist_data

random.seed(1)
np.random.seed(2)
torch.manual_seed(30)

def mcpc_landscape(ax, x_mean=1, x_var = 5, cov0 = 1, cov1 = 1):
    """
        x_mean: data mean
        x_var: data variance
        cov0: variance of input layer
        cov1: variance of latent layer
    """

    def w_dot(w, mu, x_mean, x_var):
        return (1 / ((1 + w**2)**2)) * (w * (x_var + x_mean**2) + x_mean * mu * (1 - w**2) - w * mu**2 - w - w**3)

    def mu_dot(w, mu, x_mean, x_var):
        return w * (x_mean - w * mu) / (w**2 + 1)

    def null_mu(w, x_mean):
        return x_mean / w

    def null_w_1(w, x_mean, x_var):
        numerator = (-(w**2 - 1) * x_mean + np.sqrt(((w**2 - 1) * x_mean)**2 - 4 * w * (w**3 + w * (1 - x_var - x_mean**2))))
        denominator = 2 * w
        return numerator / denominator

    def null_w_2(w, x_mean, x_var):
        numerator = (-(w**2 - 1) * x_mean - np.sqrt(((w**2 - 1) * x_mean)**2 - 4 * w * (w**3 + w * (1 - x_var - x_mean**2))))
        denominator = 2 * w
        return numerator / denominator
    
    # Define the range for w and mu with a step size of 0.01
    w = np.arange(-10, 10.01, 0.01)
    mu = np.arange(-10, 10.01, 0.01)

    # Define the ranges for w_ and mu_ with a step size of 2, and remove zeros from w_
    w_ = np.arange(-10, 11, 2)
    mu_ = np.arange(-10, 11, 2)

    W, MU = np.meshgrid(w_, mu_)

    # compute parameter flow/gradients
    W_dot = w_dot(W, MU,x_mean, x_var)
    MU_dot = mu_dot(W, MU, x_mean, x_var)

    # compute nullclines
    n_mu = null_mu(w, x_mean)
    with np.errstate(invalid='ignore'): # used to ignore RuntimeWarning: invalid value encountered in np.sqrt for w==0 (expected behavior)
        n_w_1 = null_w_1(w, x_mean, x_var)  
        n_w_2 = null_w_2(w, x_mean, x_var)  

    alpha = 0.5

    # Create a quiver plot
    scale_factor = 0.3
    ax.quiver(W[W!=0], MU[W!=0], W_dot[W!=0]* scale_factor, MU_dot[W!=0]* scale_factor, color=[0.5, 0.5, 0.5], label=r"$\Delta \theta$")

    # Plot n_mu, n_w_1, and n_w_2
    ax.plot(w[w>0], n_mu[w>0], linewidth=1.6, color=[0, 0.5, 0, alpha])
    ax.plot(w[w<0], n_mu[w<0], linewidth=1.6, color=[0, 0.5, 0, alpha], label=r'$\Delta \mu = 0$')
    ax.plot(w[w<0], n_w_1[w<0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha], label=r'$\Delta W_0 = 0$')
    ax.plot(w[w>0], n_w_1[w>0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha])
    ax.plot(w[w>0], n_w_2[w>0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha])
    ax.plot(w[w<0], n_w_2[w<0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha])

    # Scatter plot
    scatter_x = np.sqrt(x_var - 1) * np.array([1, -1])
    scatter_y = np.array([1, -1]) * x_mean / np.sqrt(x_var - 1)
    ax.scatter(scatter_x, scatter_y, color='k', linewidth=2,  facecolors='none', label='data')

    return ax

def mcpc_linear_learning(path_figures):
    # network parameters
    hidden_size = 1
    output_size = 1
    
    # sample a batch of data
    mu=1.
    var=5.
    batch_size = 256
    epochs = 3
    n = 125
    datas = [mu + np.sqrt(var)*torch.randn(batch_size, output_size) for i in range(n)]
    pseudo_input = torch.zeros(batch_size, hidden_size)
    

    # create MCPC model    
    gen_mcpc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn_normal),
        nn.Linear(hidden_size, output_size,bias=False),
    )
    gen_mcpc.train()
    
    config_mcpc = {
        "hidden_size" : 1,
        "output_size" : 1,
        "input_size" : 1,
        "input_var":1.,
        "T_pc":1,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.5},
        "mixing":150,
        "sampling":1,
        "optimizer_x_kwargs_mcpc":{"lr": 0.01},
        "optimizer_p_fn_mcpc": optim.SGD,
        "optimizer_p_kwargs_mcpc": {"lr": 0.07 , "momentum":0.2},
        "loss_fn": fe_fn 
    }

    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_mcpc, config_mcpc, training=True)

    setup_fig(zero=True)
    fig, ax = plt.subplots(figsize= (4.5, 4.))

    ax = mcpc_landscape(ax, mu, var)

    starts = [(1,7),(7,-7),(-8, 5), (-8,-4)] #  ,

    is_first=True
    for start in starts:
        print("Training MCPC with initial values: ", start)
        nn.init.constant_(gen_mcpc[0].bias,start[0]) # MCPC
        nn.init.constant_(gen_mcpc[2].weight,start[1]) 

        mcpc_weight=[start[1]]
        mcpc_mu=[start[0]]

        for e in tqdm(range(epochs)):
            for idx, data in enumerate(datas):
                mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config_mcpc["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config_mcpc["input_var"]},callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False, is_log_progress=False,is_checking_after_callback_after_t=False)
                mcpc_weight.append(gen_mcpc[2].weight[0,0].item())
                mcpc_mu.append(gen_mcpc[0].bias[0].item())

        if is_first:
            ax.plot(mcpc_weight, mcpc_mu, 'C0', linewidth=2., label="MCPC")
            is_first = False
        else:
            ax.plot(mcpc_weight, mcpc_mu, 'C0', linewidth=2.)
    
    ax.set_xlabel(r"weight $W_0$")
    ax.set_ylabel(r"prior mean $\mu$")
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_figures+ "//4b.svg")
    plt.show()

def pc_landscape(ax, x_mean=1, x_var = 5, cov0 = 1, cov1 = 1):
    """
        x_mean: data mean
        x_var: data variance
        cov0: variance of input layer
        cov1: variance of latent layer
    """

    def w_dot(w, mu, x_mean, x_var, cov0, cov1):
        return (1 / ((cov0 + cov1 * w**2)**2)) * (cov1 * w * (x_var + x_mean**2) + x_mean * mu * (cov0 - cov1 * w**2) - cov0 * w * mu**2)

    def mu_dot(w, mu, x_mean, x_var, cov0, cov1):
        return w * (x_mean - w * mu) / (cov0 + cov1 * w**2)

    def null_mu(w, x_mean):
        return x_mean / w

    def null_w_1(w, x_mean, x_var, cov0, cov1):
        numerator = -(cov0 - cov1 * w**2) * x_mean + np.sqrt(((cov0 - cov1 * w**2) * x_mean)**2 + 4 * cov0 * cov1 * w**2 * (x_var + x_mean**2))
        denominator = -2 * cov0 * w
        return numerator / denominator

    def null_w_2(w, x_mean, x_var, cov0, cov1):
        numerator = -(cov0 - cov1 * w**2) * x_mean - np.sqrt(((cov0 - cov1 * w**2) * x_mean)**2 + 4 * cov0 * cov1 * w**2 * (x_var + x_mean**2))
        denominator = -2 * cov0 * w
        return numerator / denominator
    
    # Define the range for w and mu with a step size of 0.01
    w = np.arange(-10, 10.01, 0.01)
    mu = np.arange(-10, 10.01, 0.01)

    # Define the ranges for w_ and mu_ with a step size of 2, and remove zeros from w_
    w_ = np.arange(-10, 11, 2)
    mu_ = np.arange(-10, 11, 2)

    W, MU = np.meshgrid(w_, mu_)

    # compute parameter flow/gradients
    W_dot = w_dot(W, MU,x_mean, x_var, cov0, cov1)
    MU_dot = mu_dot(W, MU, x_mean, x_var, cov0, cov1)

    # compute nullclines
    n_mu = null_mu(w, x_mean)
    n_w_1 = null_w_1(w, x_mean, x_var, cov0, cov1)
    n_w_2 = null_w_2(w, x_mean, x_var, cov0, cov1)

    alpha = 0.5

    # Create a quiver plot
    scale_factor = 0.5
    ax.quiver(W[W!=0], MU[W!=0], W_dot[W!=0]* scale_factor, MU_dot[W!=0]* scale_factor, color=[0.5, 0.5, 0.5], label=r"$\Delta \theta$")

    # Plot n_mu, n_w_1, and n_w_2
    ax.plot(w[w>0], n_mu[w>0], linewidth=1.6, color=[0, 0.5, 0, alpha])
    ax.plot(w[w<0], n_mu[w<0], linewidth=1.6, color=[0, 0.5, 0, alpha], label=r'$\Delta \mu=0$')
    ax.plot(w, n_w_1, linewidth=1.6, color=[0.8, 0.6, 1.0, alpha], label=r'$\Delta W_0 =0$')
    ax.plot(w[w>0], n_w_2[w>0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha])
    ax.plot(w[w<0], n_w_2[w<0], linewidth=1.6, color=[0.8, 0.6, 1.0, alpha])

    # Scatter plot
    scatter_x = np.sqrt(x_var - 1) * np.array([1, -1])
    scatter_y = np.array([1, -1]) * x_mean / np.sqrt(x_var - 1)
    ax.scatter(scatter_x, scatter_y, color='k', linewidth=2,  facecolors='none', label='data')

    return ax

def pc_linear_learning(path_figures):
    # network parameters
    hidden_size = 1
    output_size = 1
    

    # sample a batch of data
    mu=1.
    var=5.
    batch_size = 256
    epochs = 3
    n = 300
    datas = [mu + np.sqrt(var)*torch.randn(batch_size, output_size) for i in range(n)]
    pseudo_input = torch.zeros(batch_size, hidden_size)
    

    # create MCPC model    
    gen_pc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn_normal),
        nn.Linear(hidden_size, output_size,bias=False),
    )
    gen_pc.train()
    
    config_pc = {
        "hidden_size" : 1,
        "output_size" : 1,
        "input_size" : 1,
        "input_var":1.,
        "T_pc":150,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.1},
        "optimizer_p_fn": optim.SGD,
        "optimizer_p_kwargs": {"lr": 0.4, "momentum":0.1},
        "loss_fn": fe_fn 
    }

    # create MCPC trainer
    pc_trainer = get_pc_trainer(gen_pc, config_pc, is_mcpc=False, training=True)

    setup_fig(zero=True)
    fig, ax = plt.subplots(figsize= (4.5, 4.))

    ax = pc_landscape(ax, mu, var)

    starts = [(-8,-4),(1,7),(-8, 5), (7,-7)] #  ,

    is_first=True
    for start in starts:
        print("Training PC with initial values: ", start)
        nn.init.constant_(gen_pc[0].bias,start[0]) # MCPC
        nn.init.constant_(gen_pc[2].weight,start[1]) 

        mcpc_weight=[start[1]]
        mcpc_mu=[start[0]]

        for e in tqdm(range(epochs)):
            for idx, data in enumerate(datas):
                pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=fe_fn,loss_fn_kwargs={'_target': data, '_var': config_pc["input_var"]}, is_log_progress=False, is_checking_after_callback_after_t=False)
                mcpc_weight.append(gen_pc[2].weight[0,0].item())
                mcpc_mu.append(gen_pc[0].bias[0].item())

        if is_first:
            ax.plot(mcpc_weight, mcpc_mu, 'r', linewidth=2., label="PC")
            is_first = False
        else:
            ax.plot(mcpc_weight, mcpc_mu, 'r', linewidth=2.)
    
    ax.set_xlabel(r"weight $W_0$")
    ax.set_ylabel(r"prior mean $\mu$")
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_figures+ "//4c.svg")
    plt.show()

def comparison_linear_model(path_figures):
    # network parameters
    hidden_size = 1
    output_size = 1
    batch_size = 256
    epochs = 3
    n = 125

    start = [-7,-5] 

    # sample a batch of data
    mu=1.
    var=5.
    datas = [mu + np.sqrt(var)*torch.randn(batch_size, output_size) for i in range(n)]
    pseudo_input = torch.zeros(batch_size, hidden_size)
    
    # create PC model
    gen_pc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn_normal),
        nn.Linear(hidden_size, output_size,bias=False),
    )
    gen_pc.train()

    # create MCPC model    
    gen_mcpc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn_normal),
        nn.Linear(hidden_size, output_size,bias=False),
    )
    gen_mcpc.train()
    
    config_pc = {
        "hidden_size" : 1,
        "output_size" : 1,
        "input_size" : 1,
        "input_var":1.,
        "T_pc":150,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.5},
        "optimizer_p_fn": optim.Adam,
        "optimizer_p_kwargs": {"lr": 0.15},
        "loss_fn": fe_fn 
    }

    config_mcpc = {
        "hidden_size" : 1,
        "output_size" : 1,
        "input_size" : 1,
        "input_var":1.,
        "T_pc":1,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.5},
        "mixing":199,
        "sampling":1,
        "optimizer_x_kwargs_mcpc":{"lr": 0.005},
        "optimizer_p_fn_mcpc": optim.Adam,
        "optimizer_p_kwargs_mcpc": {"lr": 0.07},
        "loss_fn": fe_fn 
    }

    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config_pc, is_mcpc=False, training=True)
    pc_trainer_mcpc = get_pc_trainer(gen_mcpc, config_mcpc, is_mcpc=True, training=True)
    
    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_mcpc, config_mcpc, training=True)

    setup_fig(zero=True)

    nn.init.constant_(gen_pc[0].bias,start[0]) # PC
    nn.init.constant_(gen_pc[2].weight,start[1]) 
    nn.init.constant_(gen_mcpc[0].bias,start[0]) # MCPC
    nn.init.constant_(gen_mcpc[2].weight,start[1]) 

    for e in range(epochs):
        for idx, data in enumerate(tqdm(datas)):
            pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=fe_fn,loss_fn_kwargs={'_target': data, '_var': config_pc["input_var"]}, is_log_progress=False)
            pc_trainer_mcpc.train_on_batch(inputs=pseudo_input, loss_fn=fe_fn,loss_fn_kwargs={'_target': data, '_var': config_pc["input_var"]}, is_log_progress=False)
            mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config_mcpc["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config_mcpc["input_var"]},callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False, is_log_progress=False)

    ## plot samples from generative models
    num_samples = 15000
    pc_samples = sample_pc(num_samples, gen_pc, config_pc)
    mcpc_samples = sample_pc(num_samples, gen_mcpc, config_mcpc)
    
    # get data distributions
    y = np.linspace(-10,10,500)
    gen_pdf = np.exp(-0.5 * (y-mu)**2 / var)/np.sqrt(2*np.pi*var)
    
    plt.figure()
    plt.plot(y,gen_pdf,'k',label=r"$p(y)$",linewidth=3)
    plt.hist(mcpc_samples.numpy(),bins=20,density=True,label="MCPC")
    plt.hist(pc_samples.numpy(),bins=20,density=True,label="PC", color="r", alpha=0.6)
    plt.legend()
    plt.xlabel("$x_0$, y")
    plt.ylabel("probability density " + r"$p(x_0;\theta)$")
    plt.yticks([0, 0.05, 0.1, 0.15])
    plt.xlim([-12,12])
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(path_figures+ "//4a.svg")
    plt.show()

def image_reconstruction(path_models, path_figures):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config_mcpc = {
        #
        "input_size": 10,
        "hidden_size": 256,
        "hidden2_size": 256,
        "output_size": 784,
        "loss_fn": bernoulli_fn ,
        "activation_fn": 'relu',
        "input_var":None,
        #
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.7},
        #
        "mixing":50,
        "sampling":100,
        "optimizer_x_kwargs_mcpc":{"lr": 0.03},
    }

    config_pc = {
        #
        "batch_size_train":1024,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        #
        "input_size": 30,
        "hidden_size": 256,
        "hidden2_size": 256,
        "output_size": 784,
        "loss_fn": bernoulli_fn,
        "activation_fn": 'tanh',
        "input_var":None,
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.7},
    }

    config_dlgm = {
        "input_dim":784,
        "hidden_dim":256,
        "latent_dim":20,
    }

    # create model
    model_mcpc = path_models + "/mcpc_mse_1"
    gen_mcpc = get_model(config_mcpc, use_cuda)
    gen_mcpc.load_state_dict(torch.load(model_mcpc),strict=False)
    gen_mcpc.train()
    loss_fn_mcpc = fe_fn_mask if config_mcpc["loss_fn"]==fe_fn else bernoulli_fn_mask
    model_pc = "models/pc_mse_1"
    gen_pc = get_model(config_pc, use_cuda)
    gen_pc.load_state_dict(torch.load(model_pc),strict=False)
    gen_pc.train()
    loss_fn_pc = fe_fn_mask if config_pc["loss_fn"]==fe_fn else bernoulli_fn_mask
    # get dlgm 
    model_name = path_models +"/dlgm_mse_1"
    dlgm = DLGM(config_dlgm["input_dim"], config_dlgm["hidden_dim"], config_dlgm["latent_dim"], use_cuda=use_cuda, factor_recog = 1)
    dlgm.load_state_dict(torch.load(model_name),strict=False)
    
    # get data
    train_loader, val_loader, test_loader = get_mnist_data(config_pc)
    data, label = next(iter(test_loader))

    # create trainer
    trainer_mcpc = get_pc_trainer(gen_mcpc, config_mcpc, training=False, is_mcpc=True)
    trainer_pc = get_pc_trainer(gen_pc, config_pc, training=False, is_mcpc=True)

    pseudo_input_mc = torch.zeros(data.shape[0], config_mcpc["input_size"])
    pseudo_input_pc = torch.zeros(data.shape[0], config_pc["input_size"])
    if use_cuda:
        pseudo_input_mc, pseudo_input_pc, data = pseudo_input_mc.cuda(), pseudo_input_pc.cuda(), data.cuda()

    trainer_mcpc.train_on_batch(inputs=pseudo_input_mc, loss_fn=loss_fn_mcpc,loss_fn_kwargs={'_target': data}, is_log_progress=True, is_return_results_every_t=False, is_checking_after_callback_after_t=False)
    trainer_pc.train_on_batch(inputs=pseudo_input_pc, loss_fn=loss_fn_pc,loss_fn_kwargs={'_target': data}, is_log_progress=True, is_return_results_every_t=False, is_checking_after_callback_after_t=False)

    # dlgm inference
    data = data.to(dlgm.device)
    data[:,:-round(data.shape[1]/2)] = 0.
    mu, R = dlgm.recognition_model(data.view(-1, 28*28))
    img_dlgm = dlgm.generative_model(mu)

    # generate reconstructed image from latent state
    img_pc = gen_pc[-1](gen_pc[-2](gen_pc[-3].get_x().detach()))
    img_mc = gen_mcpc[-1](gen_mcpc[-2](gen_mcpc[-3].get_x().detach()))

    img_pc = img_pc.sigmoid() if config_pc["loss_fn"] ==bernoulli_fn else img_pc
    img_mc = img_mc.sigmoid() if config_mcpc["loss_fn"] ==bernoulli_fn else img_mc
            
    img_pc[:,-round(data.shape[1]/2):] = data[:,-round(data.shape[1]/2):]
    img_mc[:,-round(data.shape[1]/2):] = data[:,-round(data.shape[1]/2):]
    img_dlgm[:,-round(data.shape[1]/2):] = data[:,-round(data.shape[1]/2):]
    
    f, axs = plt.subplots(4, 10, sharey =True, sharex =True)
    for i in range(10): 
        idx=5
        tmp_pc = img_pc[label==i][idx].reshape(28,28).detach().cpu().numpy()
        tmp_mc = img_mc[label==i][idx].reshape(28,28).detach().cpu().numpy()
        tmp_dlgm = img_dlgm[label==i][idx].reshape(28,28).detach().cpu().numpy()
        tmp_d = data[label==i][idx].reshape(28,28).detach().cpu().numpy()
        tmp_d[:,:-round(data.shape[1]/2)]= 0.
        axs[0][i].imshow(tmp_d, cmap='gray')
        axs[1][i].imshow(tmp_pc, cmap='gray')
        axs[2][i].imshow(tmp_mc, cmap='gray')
        axs[3][i].imshow(tmp_dlgm, cmap='gray')
    axs[0][0].set_ylabel("input")
    axs[1][0].set_ylabel("PC")
    axs[2][0].set_ylabel("MCPC")
    axs[3][0].set_ylabel("DLGM")
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    plt.savefig(path_figures + "//4e.svg")
    plt.show()
    
def image_generation(path_models, path_figures):
    use_cuda = False

    config_pc = {
        "batch_size_train":7,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn,
        "activation_fn": 'relu',
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.1},
    }

    config_dlgm = {
        "input_dim":784,
        "hidden_dim":256,
        "latent_dim":20,
        "batch_size_train": 64,
        "batch_size_val": 1024,
        "batch_size_test": 1024
    }

    model_pc = "models/pc_fid_1"
    gen_pc = get_model(config_pc, use_cuda)
    gen_pc.load_state_dict(torch.load(model_pc),strict=False)
    gen_pc.train()
    model_name = path_models +"/dlgm_fid_1"
    dlgm = DLGM(config_dlgm["input_dim"], config_dlgm["hidden_dim"], config_dlgm["latent_dim"], use_cuda=use_cuda, factor_recog = 1)
    dlgm.load_state_dict(torch.load(model_name),strict=False)

    ## plot samples from generative models
    num_samples = 5000
    pc_samples = sample_pc(num_samples, gen_pc, config_pc, is_return_hidden=True, use_cuda=use_cuda).sigmoid_().cpu().numpy().reshape(-1,28,28)
    # mcpc_samples = sample_pc(num_samples, gen_mcpc, config_mcpc, is_return_hidden=True, use_cuda=use_cuda).sigmoid_().cpu().numpy().reshape(-1,28,28)
    dlgm_samples = dlgm.generate_samples(num_samples, is_return_hidden=True).cpu().numpy().reshape(-1,28,28)
    
    n = 8
    f, axs = plt.subplots(2, n, sharey =True, sharex =True)
    for i in range(n): 
        mult=30
        axs[0][i].imshow(pc_samples[mult*i], cmap='gray')
        axs[1][i].imshow(dlgm_samples[mult*i], cmap='gray')
    axs[0][0].set_ylabel("PC")
    axs[1][0].set_ylabel("DLGM")
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    plt.savefig(path_figures+ "//4d.svg")
    plt.show()

if __name__ == "__main__":
    pwd = os.getcwd()
    path_models = pwd + "//models"
    path_figures = pwd + "//figures"

    comparison_linear_model(path_figures)
    mcpc_linear_learning(path_figures)
    pc_linear_learning(path_figures)

    image_reconstruction(path_models, path_figures)
    image_generation(path_models, path_figures)