import numpy as np
import torch
import torch.optim as optim
import random, os

from Deep_Latent_Gaussian_Models.DLGM import DLGM
from utils.model import  get_model, bernoulli_fn
from utils.training_evaluation import get_fid, get_marginal_likelihood, get_mse_rec
from utils.data import get_mnist_data


random.seed(1)
np.random.seed(2)
torch.manual_seed(30)


def get_models_fids(path_models):
    mcpc_model_names=["mcpc_fid_1","mcpc_fid_2","mcpc_fid_3"]
    pc_model_names = ["pc_fid_1", "pc_fid_2", "pc_fid_3"]
    dlgm_model_names=["dlgm_fid_1", "dlgm_fid_2", "dlgm_fid_3"]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config_mcpc = {
        "batch_size_train":256,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        #
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn ,
        "activation_fn": 'relu',
        #
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.7},
        #
        "mixing":50,
        "sampling":100,
        "optimizer_x_kwargs_mcpc":{"lr": 0.1},
    }

    config_pc = {
        "batch_size_train":128,
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

    fids=np.zeros((3,3))
    for idx, (mcpc_name, pc_name, dlgm_name) in enumerate(zip(mcpc_model_names, pc_model_names, dlgm_model_names)):
        # get model
        model_path = path_models + "/" + mcpc_name
        gen_mcpc = get_model(config_mcpc, use_cuda)
        gen_mcpc.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=False)
        gen_mcpc.train()
        
        model_path = path_models + "/" + pc_name
        gen_pc = get_model(config_pc, use_cuda)
        gen_pc.load_state_dict(torch.load(model_path),strict=False)
        gen_pc.train()

        model_name = path_models +"/" + dlgm_name
        dlgm = DLGM(config_dlgm["input_dim"], config_dlgm["hidden_dim"], config_dlgm["latent_dim"], use_cuda=use_cuda, factor_recog = 1)
        dlgm.load_state_dict(torch.load(model_name),strict=False)
    
        is_test=True
        fids[idx,0] = get_fid(gen_mcpc, config_mcpc, use_cuda, n_samples = 5000, is_test=is_test)
        fids[idx,1] = get_fid(gen_pc, config_pc, use_cuda, n_samples = 5000, is_test=is_test)
        fids[idx,2] = dlgm.get_fid(5000, is_test=is_test)
        print(fids)
    model_type=["MCPC", "PC", "DLGM"]
    for idx in range(len(fids)):
        print("FID ", model_type[idx], ": ", fids[:,idx].mean(), "+/-", fids[:,idx].std())

def get_models_mse(path_models):
    mcpc_model_names=["mcpc_mse_1","mcpc_mse_2","mcpc_mse_3"]
    pc_model_names = ["pc_mse_1", "pc_mse_2", "pc_mse_3"]
    dlgm_model_names=["dlgm_mse_1", "dlgm_mse_2", "dlgm_mse_3"]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config_data = {
        "batch_size_train":256,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        "loss_fn": bernoulli_fn ,
    }

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

    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config_data)

    mses=np.zeros((3,3))
    for idx, (mcpc_name, pc_name, dlgm_name) in enumerate(zip(mcpc_model_names, pc_model_names, dlgm_model_names)):
        # get model
        model_path = path_models + "/" + mcpc_name
        gen_mcpc = get_model(config_mcpc, use_cuda)
        gen_mcpc.load_state_dict(torch.load(model_path),strict=False)
        gen_mcpc.train()
        
        model_path = path_models + "/" + pc_name
        gen_pc = get_model(config_pc, use_cuda)
        gen_pc.load_state_dict(torch.load(model_path),strict=False)
        gen_pc.train()

        model_name = path_models +"/" + dlgm_name
        dlgm = DLGM(config_dlgm["input_dim"], config_dlgm["hidden_dim"], config_dlgm["latent_dim"], use_cuda=use_cuda, factor_recog = 1)
        dlgm.load_state_dict(torch.load(model_name),strict=False)
    
        mses[idx,0] = get_mse_rec(gen_mcpc, config_mcpc , test_loader, use_cuda)
        mses[idx,1] = get_mse_rec(gen_pc, config_pc, test_loader, use_cuda)
        mses[idx,2] = dlgm.get_mse_rec(test_loader)
        print(mses)
    model_type=["MCPC", "PC", "DLGM"]
    for idx in range(len(mses)):
        print("MSE ", model_type[idx], ": ", mses[:,idx].mean(), "+/-", mses[:,idx].std())

def get_models_ml(path_models):
    mcpc_model_names=["mcpc_ml_1","mcpc_ml_2","mcpc_ml_3"]
    pc_model_names = ["pc_ml_1", "pc_ml_2", "pc_ml_3"]
    dlgm_model_names=["dlgm_ml_1", "dlgm_ml_2", "dlgm_ml_3"]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU")

    config_data = {
        "batch_size_train":256,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        "loss_fn": bernoulli_fn , # needed to determine how to normalize images
    }

    config_mcpc = {
        #
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn ,
        "activation_fn": 'relu',
        "input_var":None,
        #
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.1},
        #
        "mixing":50,
        "sampling":100,
        "optimizer_x_kwargs_mcpc":{"lr": 0.03},
    }

    config_pc = {
        "input_size": 25,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": bernoulli_fn,
        "activation_fn": 'tanh',
        "input_var":None,
        "T_pc":250,
        "optimizer_x_fn_pc": optim.Adam,
        "optimizer_x_kwargs_pc":{"lr": 0.3},
    }

    config_dlgm = {
        "input_dim":784,
        "hidden_dim":128,
        "latent_dim":10,
    }

    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config_data)

    mls=np.zeros((3,3))
    for idx, (mcpc_name, pc_name, dlgm_name) in enumerate(zip(mcpc_model_names, pc_model_names, dlgm_model_names)):
        # get model
        model_path = path_models + "/" + mcpc_name
        gen_mcpc = get_model(config_mcpc, use_cuda)
        gen_mcpc.load_state_dict(torch.load(model_path),strict=False)
        gen_mcpc.train()
        
        model_path = path_models + "/" + pc_name
        gen_pc = get_model(config_pc, use_cuda)
        gen_pc.load_state_dict(torch.load(model_path),strict=False)
        gen_pc.train()

        model_name = path_models +"/" + dlgm_name
        dlgm = DLGM(config_dlgm["input_dim"], config_dlgm["hidden_dim"], config_dlgm["latent_dim"], use_cuda=use_cuda, factor_recog = 1)
        dlgm.load_state_dict(torch.load(model_name),strict=False)
    
        mls[idx,0] = get_marginal_likelihood(gen_mcpc, config_mcpc , test_loader, use_cuda, n_samples=5000)
        mls[idx,1] = get_marginal_likelihood(gen_pc, config_pc, test_loader, use_cuda, n_samples=5000)
        mls[idx,2] = dlgm.get_marginal_likelihood(test_loader, n_samples=5000)
        print(mls)
    model_type=["MCPC", "PC", "DLGM"]
    for idx in range(len(mls)):
        print("FID ", model_type[idx], ": ", mls[:,idx].mean(), "+/-", mls[:,idx].std())


if __name__ == "__main__":
    pwd = os.getcwd()
    path_models = pwd + "//models"
    path_figures = pwd + "//figures"

    get_models_fids(path_models)
    get_models_mse(path_models)
    get_models_ml(path_models)