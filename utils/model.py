import torch
import torch.nn as nn
import numpy as np

import predictive_coding as pc
from torch.utils.data import TensorDataset

def sample_x_fn(inputs):
    return inputs['mu'].detach().clone().uniform_(-10.,10.)

def sample_x_fn_normal(inputs):
    return torch.randn_like(inputs['mu'])

def sample_x_fn_cte(inputs):
    return 3*torch.ones_like(inputs['mu'])

def fe_fn(output, _target, _var):
    return (1/_var)*0.5*(output - _target).pow(2).sum()

def bernoulli_fn(output, _target, _var=None, _reduction="sum"):
    loss = nn.BCEWithLogitsLoss(reduction=_reduction)
    return loss(output, _target)

def fe_fn_mask(output, _target, _var, perc=0.5):
    return (1/_var)*0.5*(output[:,-round(output.shape[1]*perc):] - _target[:,-round(output.shape[1]*perc):]).pow(2).sum()

def zero_fn(output):
    return torch.tensor(0.)


def bernoulli_fn_mask(output, _target, _var=None, perc=0.5):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    return loss(output[:,-round(output.shape[1]*perc):], _target[:,-round(output.shape[1]*perc):])

def random_step(t,_pc_trainer, var=2.):
    """
    var: needs to be 2. for mathematically correct learning.
    """
    xs = _pc_trainer.get_model_xs()
    optimizer = _pc_trainer.get_optimizer_x()
    # optimizer.zero_grad()
    for x in xs:
        x.grad.normal_(0.,np.sqrt(var/optimizer.defaults['lr']))
    optimizer.step()


def get_model(config, use_cuda, sample_x_fn = sample_x_fn):
    # create model
    if config['activation_fn'] == 'relu':
        activation_fn = nn.ReLU
    elif config['activation_fn'] == 'tanh':
        activation_fn = nn.Tanh

    gen_pc = nn.Sequential(
        nn.Linear(config["input_size"],config["input_size"]),
        pc.PCLayer(sample_x_fn=sample_x_fn),
        activation_fn(),
        nn.Linear(config["input_size"], config["hidden_size"]),
        pc.PCLayer(sample_x_fn=sample_x_fn),
        activation_fn(),
        nn.Linear(config["hidden_size"], config["hidden2_size"]),
        pc.PCLayer(sample_x_fn=sample_x_fn),
        activation_fn(),
        nn.Linear(config["hidden2_size"], config["output_size"]),
    )
    gen_pc.train()
    if use_cuda:
        gen_pc.cuda()
    return gen_pc

def get_representations(gen_pc, config, trainers, loader, rep_type="MAP", use_cuda=False, n=None):
    # rep_type:
    #   "MAP": maximum or pc inference
    #   "full": all the sampled data points
    #   "expectation": E{sampled}
    
    representations = torch.tensor([])
    labels = torch.tensor([]).type(torch.int)
    if use_cuda:
        representations, labels = representations.cuda(), labels.cuda()

    input_size=len(gen_pc[0].bias)

    # get pc representations
    if rep_type=="MAP":
        pc_trainer = trainers[0]
        for data, label in loader:
            pseudo_input = torch.zeros(data.shape[0],input_size)
            if use_cuda:
                pseudo_input, data, label = pseudo_input.cuda(), data.cuda(), label.cuda()
            pc_trainer.train_on_batch(
                inputs=pseudo_input,
                loss_fn=config["loss_fn"],
                loss_fn_kwargs={
                    '_target': data,
                    '_var': config["input_var"]
                },
                is_log_progress=True,
                is_return_results_every_t=False,
                is_checking_after_callback_after_t=False,
            )
            representations, labels = torch.concatenate((representations,gen_pc[1].get_x()), dim=0), torch.concatenate((labels, label), dim=0)

    # get mcpc representations
    elif len(trainers)==2:
        assert (rep_type == "full") or (rep_type == "expectation")
        pc_trainer = trainers[0]
        mcpc_trainer = trainers[1]

        # get indent if n is not none
        indent = 1
        if n != None:
            indent = int(config['sampling']/n)
        else:
            n = config["sampling"]

        for data, label in loader:
            pseudo_input = torch.zeros(data.shape[0],input_size)
            if use_cuda:
                pseudo_input, data, label = pseudo_input.cuda(), data.cuda(), label.cuda()
            pc_trainer.train_on_batch(
                inputs=pseudo_input,
                loss_fn=config["loss_fn"],
                loss_fn_kwargs={
                    '_target': data,
                    '_var': config["input_var"]
                },
                is_log_progress=False,
                is_return_results_every_t=False,
                is_checking_after_callback_after_t=False,
            )
            # mc inference
            results = mcpc_trainer.train_on_batch(
                inputs=pseudo_input,
                loss_fn=config["loss_fn"],
                loss_fn_kwargs={
                    '_target': data,
                    '_var': config["input_var"]
                },
                callback_after_t=random_step,
                callback_after_t_kwargs={
                    '_pc_trainer':mcpc_trainer
                },
                is_log_progress=False,
                is_return_results_every_t=True,
                is_checking_after_callback_after_t=False,
                is_sample_x_at_batch_start=False,
                is_return_representations=True
            )
            temp = torch.tensor([])
            if use_cuda:
                temp = temp.cuda()
            for rep in results["representations"]:
                temp = torch.concatenate((temp,rep.unsqueeze(0)), dim=0)
            if rep_type == "expectation":
                representations, labels = torch.concatenate((representations,temp.mean(0).detach()), dim=0), torch.concatenate((labels, label), dim=0)
            elif rep_type == "full":
                representations, labels = torch.concatenate((representations,(temp[config['mixing']::indent,:,:]).reshape(-1,temp.shape[2])), dim=0), torch.concatenate((labels, label.repeat(n)), dim=0)
                
    else:
        raise NotImplementedError

    return TensorDataset(representations, labels)
