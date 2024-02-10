import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random 
import os
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import predictive_coding as pc

from utils.plotting import setup_fig, generate_video
from utils.model import sample_x_fn, zero_fn, random_step, get_model
from utils.training_evaluation import get_pc_trainer, get_mcpc_trainer


random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


def generation_linear_model(path_figures):
    # network parameters
    hidden_size = 1
    output_size = 1
    batch_size = 1

    config = {
        # PC parameters
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.5},
        # MCPC parameters
        "mixing":0,
        "sampling":10000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.3},
        # variance of the sensory layer 
        "input_var":1.,
    }

    # sample a batch of data
    pseudo_input = torch.zeros(batch_size, hidden_size)
    
    # create model
    print("making model")
    def gen_pc_layer_loss(inputs):
        return (1/config["input_var"])*0.5 *(inputs['mu'] - inputs['x'])**2 

    gen_pc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn),
        nn.Linear(hidden_size, output_size,bias=False),
        pc.PCLayer(energy_fn=gen_pc_layer_loss,sample_x_fn=sample_x_fn)
    )
    # initialise model
    gen_pc.train()
    nn.init.constant_(gen_pc[0].bias,0.5)
    nn.init.constant_(gen_pc[2].weight,2.) 

    print("making trainers")
    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True, training=False)
    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    print("Inference")
    # train on this batch with this loss function
    print("MAP")
    pc_trainer.train_on_batch(inputs=pseudo_input,is_log_progress=True,is_return_results_every_t=True,is_return_outputs =True)
    print("MCPC")
    # mc inference
    mc_results = mcpc_trainer.train_on_batch(inputs=pseudo_input,callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_return_results_every_t=True,is_return_outputs =True)
    
    y = np.linspace(-10,10,500)
    gen_pdf = (1/np.sqrt(2*np.pi*(2.**2+config["input_var"])))*np.exp(-0.5*(y-gen_pc[0].bias.item()*2.)**2/(2.**2+config["input_var"]))

    setup_fig(zero=True)
    # plt.hist(generated_inputs,bins=20, density=True)
    plt.plot(y,gen_pdf,'k',label=r"$p(x_0;\theta)$",linewidth=3)
    plt.hist([mc_results["outputs"][-i][0,0].item() for i in range(config["sampling"])],bins=20,density=True,label="MCPC") # no need to divide by /gen_pc[2].weight.item()
    plt.legend()
    plt.xlabel("$x_0$")
    plt.ylabel("probability density")
    plt.yticks([0, 0.05, 0.1, 0.15, 0.20])
    plt.xlim([-6,9])
    plt.ylim([0,0.22])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_figures +"//3a.svg")
    plt.show()

    data = [mc_results["outputs"][-i][0,0].item() for i in range(config["sampling"])]

    setup_fig(zero=True)
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 6))
    fig.set_figwidth(5)
    fps=50
    data_indent=2
    
    def make_frames_mc(t):
        axs.clear()
        idx_time = int(t*fps*data_indent)
        axs.hist(data[:idx_time+1], density=True, bins=np.linspace(-12,12,20), label="hist($x_0(t)$), [0, t]")
        axs.plot(y,gen_pdf,'k',label=r"$p(x_0;\theta)$",linewidth=3)
        axs.scatter(data[idx_time], 0, c="orange", s=70, label = r"x$_0$(t)")
        axs.set_xlabel("$x_0$")
        axs.set_ylabel("probability density")
        axs.set_xlim([-10,10])
        axs.set_ylim([-0.025, 0.3])
        plt.legend(loc=0)
        plt.tight_layout()
        return mplfig_to_npimage(axs.get_figure()) if axs is not None else mplfig_to_npimage(fig)
    duration_mc=(len(data)//data_indent)/fps
    final_clip = VideoClip(make_frames_mc, duration=duration_mc)

    final_clip.write_gif(path_figures +"//3a.gif",fps=fps)


def generation_non_linear_model(path_figures, path_models):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config = {
        "batch_size_train":64,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        # model architecture
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "activation_fn": 'relu',
        "loss_fn": zero_fn, 
        # PC
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.7},
        # MCPC
        "mixing":1000,
        "sampling":30000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.1},
    }

    # create model
    model_name = path_models + "/mcpc_fid_3"
    gen_pc = get_model(config, use_cuda)
    gen_pc.load_state_dict(torch.load(model_name),strict=False)
    gen_pc.train()

    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config, training=False, is_mcpc=True)
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    N=1
    pseudo_input = torch.zeros(N,config["input_size"])
    if use_cuda:
        pseudo_input = pseudo_input.cuda()
    pc_trainer.train_on_batch(inputs=pseudo_input, is_log_progress=True, is_return_results_every_t=True)
    mc_results = mcpc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=config["loss_fn"], loss_fn_kwargs={}, callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=False,is_log_progress=True,is_return_results_every_t=True,is_checking_after_callback_after_t=False, is_return_outputs=True)
            
    ims = [img.reshape(28,28).detach().sigmoid_().cpu().numpy() for img in mc_results["outputs"]]
    # ims = ims[:]
    # get samples for figure
    nrow=2
    ncol=5
    f, axs = plt.subplots(nrow,ncol, sharey=True)
    indent= 3000
    for i in range(nrow*ncol):
        axs[int(i/ncol), i%ncol].imshow(ims[config["mixing"] + i*indent], cmap='gray')
        axs[int(i/ncol)][i%ncol].axis("off")
    plt.suptitle("Generated with sampler")
    plt.savefig(path_figures +"//3b_and_4d.svg")
    plt.show()

    # make animation
    data_indent = 20
    generate_video(ims[::data_indent], show=True, save=True, title="input neuron activity", file_name="3b_and_4d")



if __name__ == "__main__":
    pwd = os.getcwd()
    path_models = pwd + "//models"
    path_figures = pwd + "//figures"

    generation_linear_model(path_figures)
    generation_non_linear_model(path_figures, path_models)