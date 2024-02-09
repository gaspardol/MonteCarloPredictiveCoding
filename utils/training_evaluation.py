from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import tempfile, shutil, os, subprocess


import predictive_coding as pc
from utils.model import bernoulli_fn, fe_fn, bernoulli_fn_mask, fe_fn_mask
from utils.data import make_compressed_MNIST_files

def get_pc_trainer(gen_pc, config, is_mcpc=False, training=True):
    if is_mcpc:
        pc_trainer = pc.PCTrainer(gen_pc, 
            T=config["T_pc"], 
            update_x_at='all', 
            optimizer_x_fn=config["optimizer_x_fn_pc"],
            optimizer_x_kwargs=config["optimizer_x_kwargs_pc"],
            early_stop_condition = "False",
            update_p_at="never",   
            plot_progress_at=[]
        )
    else:
        pc_trainer = pc.PCTrainer(gen_pc, 
            T=config["T_pc"], 
            update_x_at='all', 
            optimizer_x_fn=config["optimizer_x_fn_pc"],
            optimizer_x_kwargs=config["optimizer_x_kwargs_pc"],
            early_stop_condition = "False",
            update_p_at= "last" if training else "never",   
            optimizer_p_fn=config["optimizer_p_fn"],
            optimizer_p_kwargs=config["optimizer_p_kwargs"],
            plot_progress_at=[]
        )
    return pc_trainer



def get_mcpc_trainer(gen_pc, config, training=True):
    mcpc_trainer = pc.PCTrainer(
        gen_pc,
        T=config["mixing"]+config["sampling"],
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs=config["optimizer_x_kwargs_mcpc"],
        update_p_at="last" if training else "never",
        accumulate_p_at=[i+config["mixing"] for i in range(config["sampling"])],
        optimizer_p_fn= config["optimizer_p_fn_mcpc"] if training else optim.SGD,
        optimizer_p_kwargs=config["optimizer_p_kwargs_mcpc"] if training else {"lr": 0.0},
        plot_progress_at=[]
    )
    return mcpc_trainer

def get_mcpc_trainer_one_sample(gen_pc, config, training=True):
    mcpc_trainer = pc.PCTrainer(
        gen_pc,
        T=config["K"],
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs=config["optimizer_x_kwargs_mcpc"],
        update_p_at="last" if training else "never",
        optimizer_p_fn= config["optimizer_p_fn_mcpc"] if training else optim.SGD,
        optimizer_p_kwargs=config["optimizer_p_kwargs_mcpc"] if training else {"lr": 0.0},
        plot_progress_at=[]
    )
    return mcpc_trainer

def sample_pc(num_samples, model, config, use_cuda=False, is_return_hidden=False):
    def sample_multivariate_Gauss(mean, cov, use_cuda=False):
        L = torch.linalg.cholesky(cov)
        rand = torch.randn((mean.shape[0],mean.shape[1], 1))
        rand = torch.matmul(L, rand).view(mean.shape)
        if use_cuda:
            rand = rand.cuda()
        return (mean + rand).detach()
    
    temp = torch.zeros((num_samples,config["input_size"]))
    if use_cuda:
        temp = temp.cuda()

    for layer_idx in range(len(model)):
        if isinstance(model[layer_idx], pc.PCLayer):
            temp = sample_multivariate_Gauss(temp,torch.eye(temp.shape[1]), use_cuda=use_cuda)
        else:
            temp = model[layer_idx](temp)
    
    if is_return_hidden:
        return temp.detach()

    if config["loss_fn"] == fe_fn:
        temp = sample_multivariate_Gauss(temp, config["input_var"]*torch.eye(temp.shape[1]), use_cuda=use_cuda)
    elif config["loss_fn"] == bernoulli_fn:
        temp = temp.sigmoid()
        temp = (torch.rand_like(temp)<=temp).double()

    return temp.detach()
    


def get_fid(gen_pc, config, use_cuda, n_samples = 5000, is_test=False):
    # check if MNIST_data/MNIST/test_img.npz exists
    if (not os.path.isfile("MNIST_data/MNIST/test_img.npz")) or (not os.path.isfile("MNIST_data/MNIST/val_img.npz")):
        print("MNIST_data/MNIST/test_img.npz or MNIST_data/MNIST/test_img.npz does not exist")
        print("Creating MNIST_data/MNIST/test_img.npz and MNIST_data/MNIST/val_img.npz")
        make_compressed_MNIST_files()

    samples = sample_pc(n_samples, gen_pc, config, use_cuda=use_cuda, is_return_hidden=True)
    images = samples.view(-1,28,28)
    if config["loss_fn"] == fe_fn:
        images = (images > 0).type_as(images)
    if config["loss_fn"] == bernoulli_fn:
        images = images.sigmoid()

    tf = tempfile.NamedTemporaryFile()
    
    new_folder = False
    while not new_folder:
        try:
            new_folder=True
            os.makedirs(tf.name+"_")
            print(tf.name+"_")
        except OSError:
            print("ERROR")
            tf = tempfile.NamedTemporaryFile()
            new_folder=False
    
    for img_idx in range(len(images)):
        save_image(images[img_idx], tf.name+"_"+"\\"+str(img_idx)+".png")

    if is_test:
        result = subprocess.run('python -m pytorch_fid MNIST_data/MNIST/test_img.npz '+ tf.name+"_", stdout=subprocess.PIPE)
    else:
        result = subprocess.run('python -m pytorch_fid MNIST_data/MNIST/val_img.npz '+ tf.name+"_", stdout=subprocess.PIPE)
    shutil.rmtree(tf.name+"_")
    return float(str(result.stdout).split(" ")[2].split("\\r")[0])



def get_mse_rec(gen_pc, config , dataloader, use_cuda):

    if config["loss_fn"]==fe_fn:
        loss_fn = fe_fn_mask
    elif config["loss_fn"]==bernoulli_fn:
        loss_fn = bernoulli_fn_mask

    gen_pc.train()

    # create trainer
    pc_trainer = get_pc_trainer(gen_pc, config, training=False, is_mcpc=True)
    
    MSE, n_data = 0., 0
    for data, _ in dataloader:
        # warm up set correct batch_size in pc activaitons
        pseudo_input = torch.zeros(data.shape[0],config["input_size"])
        if use_cuda:
            pseudo_input, data = pseudo_input.cuda(), data.cuda()
        # get MAP
        pc_trainer.train_on_batch(inputs=pseudo_input, loss_fn=loss_fn,loss_fn_kwargs={'_target': data, '_var': config["input_var"]}, is_log_progress=False, is_return_results_every_t=False, is_checking_after_callback_after_t=False)
        img = gen_pc[-3].get_x().detach()
        img = gen_pc[-2](img)
        img = gen_pc[-1](img)
        
        if config["loss_fn"]==bernoulli_fn:
            # threshold with 0 because no sigmoid was applied
            img = (img>0).type_as(img)

        # find MSE
        MSE += ((img[:,:-round(data.shape[1]/2)] - data[:,:-round(data.shape[1]/2)])**2).mean(1).sum()
        n_data+= data.shape[0]
    return MSE/n_data


def get_marginal_likelihood(gen_pc, config , dataloader, use_cuda, n_samples = 5000):    
    # get latent state x_1 and reshape to [1, n_samples, latent_size]
    latent_samples = sample_pc(n_samples, gen_pc, config, use_cuda=use_cuda, is_return_hidden=True).cpu().unsqueeze(0)
    latent_samples = torch.clamp(latent_samples, -20, 20)

    loss = nn.BCEWithLogitsLoss(reduction='none')
    # make appropriate dataloader
    dataloader_ml = DataLoader(dataloader.dataset, batch_size=int(100*6000/n_samples))
    ml=0.
    losses = torch.tensor([])

    for data, _ in tqdm(dataloader_ml):
        batch_size = data.shape[0]
        # reshape batch to [batch_size, 1, data_size(=latent_size)]
        data = data.unsqueeze(1)
        if config["loss_fn"]==fe_fn:
            latent_samples_, data = latent_samples.numpy(), data.numpy()
            p = (2*np.pi* config["input_var"])**(-0.5*data.shape[-1])*np.exp(-0.5*((data-latent_samples_)**2).sum(-1))
            ml += np.log(p.mean(1)).sum(0)
            raise NotImplementedError
        elif config["loss_fn"]==bernoulli_fn:
            # torch.sigmoid_(latent_samples)
            data = data.repeat(1, n_samples, 1)
            latent_samples_ = latent_samples.repeat(batch_size,1,1)
            l = loss(latent_samples_, data).view(batch_size, n_samples, latent_samples_.shape[-1]).sum(-1)
            losses = torch.concatenate((losses,l), dim=0)
    m_loss = losses.min(1).values #(losses.max()-losses.min())/2
    p = torch.exp(-(losses-m_loss.unsqueeze(1))).mean(1)
    ml += (np.log(p) - m_loss).mean()
    return ml

def train(model, x, y, optimizer, criterion):
    # model.zero_grad()
    output = model(x)
    optimizer.zero_grad()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output

def test(model, testloader, print_acc=False):
    correct_count, all_count = 0., 0.
    for data, labels in testloader:
        pred = torch.max(torch.exp(model(data)), 1)
        correct = (pred.indices == labels).long()
        correct_count += correct.sum()
        all_count += correct.size(0)
    acc =correct_count / all_count
    if print_acc:
        print("Model Accuracy =", acc)
    return acc

class MNIST_LinearClassifier(nn.Module):
    def __init__(self, rep_size):
        super().__init__()
        self.lin = nn.Linear(rep_size, 10)

    def forward(self, x):
        x = self.lin(x)
        return x



def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def kl_divergence_discrete(p, q):
    """
    Computes the KL divergence between two discrete probability distributions p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Ensure that the probability distributions sum to 1.
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Compute the KL divergence.
    kl_div = np.sum(np.where(p != 0, - p * np.log(q / p), 0))
    
    return kl_div


def get_paired_stat(before, after, type="two-sided"):
    from scipy import stats
    from scipy.stats import shapiro
    # check normality
    _, p_norm = shapiro([a-m for (a,m) in zip(before, after)])

    if p_norm > 0.05:
        print("relative t-test")
        _, p = stats.ttest_rel(before, after, alternative=type)
    else:
        print("wilcoxon")
        _, p = stats.wilcoxon(before, after, alternative=type)

    return p