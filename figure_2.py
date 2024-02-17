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
from utils.model import sample_x_fn_cte, bernoulli_fn, bernoulli_fn_mask, fe_fn,fe_fn_mask, get_model, get_representations, random_step
from utils.training_evaluation import train, test, get_pc_trainer, get_mcpc_trainer
from utils.data import get_mnist_data
from utils.plotting import setup_fig, proba_to_coordinate


random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


def posterior_linear_model(path_figures):
    # network parameters
    hidden_size = 1
    output_size = 1

    # sample a batch of data
    batch_size = 1
    data = torch.ones(batch_size, output_size)
    pseudo_input = torch.zeros(batch_size, hidden_size)
    
    # create model
    gen_pc = nn.Sequential(
        nn.Linear(hidden_size,hidden_size), # input zeros ($pseudo_input) so that weight are disregarded and only biases are used
        pc.PCLayer(sample_x_fn=sample_x_fn_cte),
        nn.Linear(hidden_size, output_size,bias=False),
    )
    # initialise model
    gen_pc.train()
    nn.init.constant_(gen_pc[0].bias, 0.2)
    nn.init.constant_(gen_pc[2].weight, 2.)

    config = {
        # variance of the sensory layer 
        "input_var":1.,
        # parameters for PC inference
        "T_pc":2000,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.02},
        # parameters for MCPC inference
        "mixing":0,
        "sampling":10000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.02},
        "optimizer_p_fn_mcpc": optim.Adam,
        # sets sensory layer to Gaussian layer 
        "loss_fn": fe_fn 
    }

    # create PC trainer
    pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True, training=False)
    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    # PC inference
    pc_results = pc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target':data,'_var':config["input_var"]},is_return_results_every_t=True, is_return_representations=True)
    map = gen_pc[1].get_x()[0,0].item()
    # MCPC inference
    mc_results = mcpc_trainer.train_on_batch(inputs=pseudo_input,loss_fn=config["loss_fn"],loss_fn_kwargs={'_target': data,'_var':config["input_var"]},callback_after_t=random_step,callback_after_t_kwargs={'_pc_trainer':mcpc_trainer},is_sample_x_at_batch_start=True,is_return_results_every_t=True, is_return_representations=True)
    
    # compute true posterior for generative model using Bayes theorem
    x_post = np.linspace(-10,10,1000)
    post = np.sqrt(gen_pc[2].weight.item()**2+config["input_var"])/(np.sqrt(2*np.pi*config["input_var"])) * np.exp(-0.5*((x_post-gen_pc[0].bias.item())**2+(data.item()-gen_pc[2].weight.item()*x_post)**2/(config["input_var"])-(data.item()-gen_pc[2].weight.item()*gen_pc[0].bias.item())**2/(gen_pc[2].weight.item()**2+config["input_var"])))
   
    ## compare PC and MCPC inference to true posterior
    setup_fig(zero=False)
    plt.plot(x_post,post,'k',label=r"$p(x_1|y;\theta)$",linewidth=3)
    plt.hist([mc_results["representations"][-i][0,0].item() for i in range(config["sampling"])],bins=20,density=True,label="MCPC") # need to divide by /gen_pc[2].weight.item() because output is w*x
    plt.vlines(map, 0, 1, colors='r', label="PC", linewidth=3)
    # plt.bar(map, 1., color = 'r', width = 0.2, label="PC")
    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel("probability density")
    plt.xlim([-2,4.5])
    plt.ylim([0,1.])
    plt.yticks([0.,0.2,0.4,0.6,0.8,1.])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_figures + "//2b.svg")
    plt.show()


    # plt time evolution of activities
    setup_fig()
    plt.figure()
    plt.plot( [mc_results["representations"][i][0,0].item() for i in range(config["sampling"]+config["mixing"])],"C0", label="MCPC", linewidth=2.5)
    plt.plot( [pc_results["representations"][i][0,0].item() for i in range(config["T_pc"])], "red", label="PC", linewidth=3)
    plt.xlim([-5,1000])
    plt.ylim([-1.1, 3.1])
    plt.xlabel("time (AU)")
    plt.ylabel(r"$x_1$")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path_figures + "//2a.svg")
    plt.show()

    ## Make animation if the sampling process
    # Keep MCPC samples after mixing period and resample
    data = [mc_results["representations"][-i][0,0].item() for i in range(config["sampling"])]
    data = data[1000::5]
    # make animation
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    setup_fig(zero=True)
    fig.set_figheight(4.5)
    plt.axis('off')
    fps=50
    def make_frames_mc(t):
        axs.clear()
        idx_time = int(t*fps)
        axs.hist(data[:idx_time+1], density=True, bins=np.linspace(-1.5,2.5,20), label = r"hist($x_1(t)$), [0, t]")
        axs.plot(x_post,post,'k',label=r"$p(x_1|y;\theta)$",linewidth=3)
        axs.scatter(data[idx_time], 0, c="orange", s=70, label = r"$x_1$(t)")
        axs.set_xlabel(r"$x_1$")
        axs.set_ylabel("probability")
        axs.set_xlim([-1.5,3.5])
        axs.set_ylim([-0.1, 1.4])
        axs.legend(loc=1)
        plt.tight_layout()
        return mplfig_to_npimage(axs.get_figure()) if axs is not None else mplfig_to_npimage(fig)
    duration_mc=len(data)/fps
    final_clip = VideoClip(make_frames_mc, duration=duration_mc)
    final_clip.write_gif(path_figures + "//2b.gif",fps=fps)

    plt.show()

def posterior_non_linear_model(path_models, path_figures, img_kept=0.5):
    """
        img_kept : percentage of image kept
    """
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(30)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config = {
        "batch_size_train":1024,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        #
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn ,
        "activation_fn": 'relu',
        "input_var":None,
        #
        "T_pc":2000,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.1},
        #
        "mixing":1000,
        "sampling":9000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.03},
    }


    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config)

    # get model
    gen_pc = get_model(config, use_cuda)
    model_name=path_models + "\mcpc_ml_2" 
    gen_pc = get_model(config, use_cuda)
    gen_pc.load_state_dict(torch.load(model_name),strict=False)
    gen_pc.train()

    # generate data
    if config["loss_fn"]==fe_fn:
        loss_fn = fe_fn_mask
    elif config["loss_fn"]==bernoulli_fn:
        loss_fn = bernoulli_fn_mask

    # get trainers
    pc_trainer = get_pc_trainer(gen_pc, config, training=False, is_mcpc=True)
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    ################################################## get representations  ##########################################3
    print("getting representations")
    rep_dataset = get_representations(gen_pc, config, [pc_trainer, mcpc_trainer], train_loader, rep_type="MAP", use_cuda=use_cuda)
    representations = DataLoader(rep_dataset, batch_size=512, shuffle=False)

    ################################################## train linear classifier ######################################3
    # make classifier
    classifier = MNIST_LinearClassifier(config["input_size"])
    criterion = nn.CrossEntropyLoss()
    optm = optim.Adam(classifier.parameters(), lr=0.05) #, momentum=0.0
    if use_cuda:
        classifier.cuda()

    # train classifier
    print("training classifier")
    EPOCHS = 10
    for epoch_idx in range(EPOCHS):
        for data, label in representations:
            loss, _ = train(classifier, data, label, optm, criterion)
        test(classifier,representations,print_acc=True)

    ################################################## make masked reconstruction ######################################3
    digit = 4 # digit to analyse
    data, label = list(test_loader)[1]
    data, label = data[label==digit], label[label==digit]
    pseudo_input = torch.zeros(data.shape[0],config["input_size"])
    if use_cuda:
        pseudo_input, data, label = pseudo_input.cuda(), data.cuda(), label.cuda()

    # perform inference with masked image
    pc_results = pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=loss_fn,loss_fn_kwargs={'_target': data, '_var': config["input_var"], 'perc':img_kept}, is_log_progress=True, is_return_outputs=False, is_return_representations=True ,is_return_results_every_t=True, is_checking_after_callback_after_t=False)
    mcpc_results = mcpc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=loss_fn,loss_fn_kwargs={'_target': data, '_var': config["input_var"], 'perc':img_kept}, callback_after_t=random_step, callback_after_t_kwargs={'_pc_trainer':mcpc_trainer}, is_sample_x_at_batch_start=False, is_return_outputs=False, is_return_representations=True, is_log_progress=True, is_return_results_every_t=True)

    # map representations to class probabilities using classifier
    preds_pc = [torch.softmax(classifier(data.cuda() if use_cuda else data),1).detach().cpu() for data in pc_results["representations"]]
    preds_mcpc = [torch.softmax(classifier(data.cuda() if use_cuda else data),1).detach().cpu() for data in mcpc_results["representations"]]
    preds_mcpc = preds_mcpc[config['mixing']:]
    # convert list of tensors to 3D tensor
    preds_pc = torch.concatenate([p.view(-1,1,10) for p in preds_pc], axis=1)
    preds_mcpc = torch.concatenate([p.view(-1,1,10) for p in preds_mcpc], axis=1)
    gridsize=20
    file_type = "full" if img_kept==1. else "masked"
    for idx_batch in range(10):
        img = data[idx_batch].clone()
        img[:round(len(img)*(1-img_kept))] = 0.
        
        ## make recap plot
        # get coors
        coor_pc, _ = proba_to_coordinate(preds_pc[idx_batch, -1, :])
        coor_previous, class_coor  = proba_to_coordinate(preds_mcpc[idx_batch, :,:])
        # make plot
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        axs.set_aspect('equal')
        plt.axis('off')
        axs.hexbin(coor_previous[0],coor_previous[1], gridsize=gridsize, cmap='Blues', bins=preds_mcpc.shape[1],extent=(-1, 1, -1, 1), label="MCPC")
        for idx in range(10):
            axs.text(1.15*class_coor[0][idx]-0.038,1.15*class_coor[1][idx]-0.04,str(idx),fontsize=20)    
        axs.scatter(coor_pc[0][0],coor_pc[1][0], c="red", linewidths=6, marker="o", facecolor='none', label="PC")
        axs.set_xlim([-1.2,1.2])
        axs.set_ylim([-1.2,1.2])
        plt.legend(fontsize=14, loc=3)
        plt.savefig(path_figures + "//digit_posteriors//"+file_type +"_"+str(idx_batch)+".svg")
        if idx_batch==4:
            name = "2c" if img_kept==1. else "2d"
            plt.savefig(path_figures + "//"+name+".svg")
        plt.close()

        # save animation for figure shown in manuscipt
        if idx_batch==4:
            fps=100    
            data_indent = 15    
            fig, axs = plt.subplots(1, 1, constrained_layout=True)
            axs.set_aspect('equal')
            plt.axis('off')
            def make_frames_mc(t):
                axs.clear()
                axs.axis('off')
                idx_time = int(t*fps*data_indent)
                coor, class_coor = proba_to_coordinate(preds_mcpc[idx_batch, idx_time, :])
                coor_previous, _  = proba_to_coordinate(preds_mcpc[idx_batch, :idx_time+1,:])
                axs.hexbin(coor_previous[0],coor_previous[1], gridsize=gridsize, cmap='Blues', bins=idx_time,extent=(-1, 1, -1, 1), label=r"hist($x_L(t)$), [0, t]")
                for idx in range(10):
                    axs.text(1.15*class_coor[0][idx]-0.038,1.15*class_coor[1][idx]-0.04,str(idx),fontsize=15)    
                axs.scatter(coor[0][0],coor[1][0], c="orange", label=r"x$_L$ (t)")
                axs.set_xlim([-1.2,1.2])
                axs.set_ylim([-1.2,1.2])
                axs.legend(fontsize=14)
                title = "MCPC inference for full image" if img_kept==1 else "MCPC inference for masked image"
                axs.set_title(title, fontsize=14)
                return mplfig_to_npimage(axs.get_figure()) if axs is not None else mplfig_to_npimage(fig)
            duration_mc=preds_mcpc.shape[1]/(fps*data_indent)
            animation_mc = VideoClip(make_frames_mc, duration=duration_mc)
            name = "2c" if img_kept==1. else "2d"
            animation_mc.write_gif(path_figures + "//"+name+".gif",fps=fps)
            plt.close()

        save_image(img.view(28,28), path_figures + "//digit_posteriors//"+ file_type + "_" + str(idx_batch) +".png")
        if idx_batch==4:
            name = "2c" if img_kept==1. else "2d"
            save_image(img.view(28,28), path_figures + "//"+ name +".png")
        

def comparison_ideal_observer(path_models, path_figures):
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(30)

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    if use_cuda:
        print("Using GPU")

    config = {
        "batch_size_train":512,
        "batch_size_val": 1024,
        "batch_size_test": 128,
        #
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn ,
        "activation_fn": 'relu',
        "input_var":None,
        #
        "T_pc":2000,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.1},
        #
        "mixing":1000,
        "sampling":9000,
        "optimizer_x_kwargs_mcpc":{"lr": 0.03},
    }

    # get pc
    model_name=path_models + "\mcpc_ml_2" 
    gen_pc = get_model(config, use_cuda)
    gen_pc.load_state_dict(torch.load(model_name),strict=False)

    # get data
    train_loader, val_loader, test_loader = get_mnist_data(config)

    # create trainer for MAP inference
    pc_trainer = get_pc_trainer(gen_pc, config, is_mcpc=True, training=False)
    # create MCPC trainer
    mcpc_trainer = get_mcpc_trainer(gen_pc, config, training=False)

    ################################################## get representaitons  ##########################################3
    print("getting representaitons")
    rep_dataset = get_representations(gen_pc, config, [pc_trainer, mcpc_trainer], train_loader, rep_type="MAP", use_cuda=use_cuda)
    representations = DataLoader(rep_dataset, batch_size=512, shuffle=False)

    ################################################## train linear classifier ######################################3
    # make classifier
    classifier = MNIST_LinearClassifier(config["input_size"])
    criterion = nn.CrossEntropyLoss()
    optm = optim.Adam(classifier.parameters(), lr=0.05) #, momentum=0.0
    if use_cuda:
        classifier.cuda()

    # train classifier
    print("training classifier")
    EPOCHS = 10
    for epoch_idx in range(EPOCHS):
        for data, label in representations:
            train(classifier, data, label, optm, criterion)
        test(classifier, representations,print_acc=True)
    del representations, rep_dataset
    ####################################### get full image posteriors #################################################

    loss_fn = bernoulli_fn_mask if config["loss_fn"]==bernoulli_fn else fe_fn_mask

    kls = np.zeros((4,1))
    for data, label in test_loader:
        pseudo_input = torch.zeros(data.shape[0],config["input_size"])
        if use_cuda:
            pseudo_input, data, label = pseudo_input.cuda(), data.cuda(), label.cuda()

        # PC inference
        pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=loss_fn, loss_fn_kwargs={'_target': data, '_var':config["input_var"]})
        pc_representation = pc_trainer.get_model_representations()
        probability_pc = torch.softmax(classifier(pc_representation), 1).detach().cpu()
        probability_pc = probability_pc + 10**(-4)
        probability_pc /= probability_pc.sum(1).unsqueeze(1) 

        # mc inference
        results = mcpc_trainer.train_on_batch( inputs=pseudo_input, loss_fn=loss_fn, loss_fn_kwargs={ '_target': data, '_var':config["input_var"]},
            callback_after_t=random_step, callback_after_t_kwargs={ '_pc_trainer':mcpc_trainer},
            is_log_progress=True, is_return_results_every_t=True, is_checking_after_callback_after_t=False, is_sample_x_at_batch_start=False, is_return_representations=True
        )
        mcpc_representation = results["representations"]

        probability_mcpc = torch.zeros(mcpc_representation[0].shape[0], 10)
        digit_infered = torch.zeros(config["sampling"], mcpc_representation[0].shape[0])
        for idx in tqdm(range(config["sampling"])):
            representation = mcpc_representation[idx+config["mixing"]].cuda() if use_cuda else mcpc_representation[idx+config["mixing"]]
            probability_mcpc += torch.softmax(classifier(representation), 1).detach().cpu()
            digit_infered[idx] = torch.argmax(probability_mcpc, dim=-1)
        probability_mcpc /= config["sampling"]
        probability_mcpc = probability_mcpc + 10**(-4)
        probability_mcpc /= probability_mcpc.sum(1).unsqueeze(1) 

        del mcpc_representation, pc_representation
        # get CCN probability
        name = path_models +"/resnet9"
        resnet = ResNet9().cuda()
        resnet.load_state_dict(torch.load(name),strict=False)
        data_cnn = data.view(-1,1,28,28)
        data_cnn[:,:,:14,:] = 0
        probability_cnn = torch.softmax(resnet(data_cnn).cpu(), 1).detach().cpu()

        # compute KL divergence
        kls[0] += kl_divergence_discrete(probability_cnn, probability_mcpc)
        kls[1] += kl_divergence_discrete(probability_cnn, probability_pc)
        kls[2] += kl_divergence_discrete(probability_cnn, probability_mcpc[np.random.permutation(len(probability_mcpc))]) 
        kls[3] += kl_divergence_discrete(probability_cnn, probability_pc[np.random.permutation(len(probability_pc))]) 
        
    results = pd.DataFrame(data={"KL":[kls[0,0], kls[1,0], kls[2,0], kls[3,0]]}, index=["MCPC", "PC", "MC shuffled", "PC shuffled"])
    print(results)

    setup_fig()
    xlabel = ["MCPC", "PC", "random"]
    colors = ["C0", "r", "grey"]
    kls_plot = kls[[0,1,2],:].squeeze()
    kls_plot[2] = kls[2:,:].mean()
    bar = plt.bar(xlabel, kls_plot, width=0.6)
    for b,c in zip(bar,colors):
        b.set_color(c)
    plt.yticks([0,30,60,150])
    plt.ylabel("KL divergence")
    plt.tight_layout()
    plt.xticks(xlabel, labels=xlabel)
    plt.savefig(path_figures + "//2e.svg")
    plt.show()



if __name__ == "__main__":
    pwd = os.getcwd()
    path_models = pwd + "//models"
    path_figures = pwd + "//figures"

    posterior_linear_model(path_figures)
    posterior_non_linear_model(path_models, path_figures, img_kept=0.5)
    posterior_non_linear_model(path_models, path_figures, img_kept=1.)
    comparison_ideal_observer(path_models, path_figures) 