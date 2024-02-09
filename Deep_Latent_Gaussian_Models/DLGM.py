import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
import tempfile, shutil
import os, subprocess
from tqdm import tqdm
from itertools import chain


from utils.training_evaluation import test, MNIST_LinearClassifier
from utils.data import make_compressed_MNIST_files


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

class RankOneFactor:
    def __init__(self, size, delta=1e-6):
        self.size = size
        self._free_parameter_size = 2*size

        self.delta = delta

        self.diag_ii = torch.arange(self.size)
        self.diag_jj = torch.arange(self.size)

    def free_parameter_size(self):
        return self._free_parameter_size

    def parameterize(self, free_parameter):
        '''
        batch_size x free_parameter_size -> batch_size x size x size
        '''
        batch_size = free_parameter.shape[0]

        assert free_parameter.shape[1] == self.free_parameter_size()
        R = torch.zeros(batch_size, self.size, self.size, device=free_parameter.device)
        v = free_parameter[:, self.size:].exp() + self.delta
        R[:, self.diag_ii, self.diag_jj] = v
        R += torch.einsum('bi,bj->bij', v, v)
        R[:, self.diag_ii, self.diag_jj] -= v
        R[:, self.diag_ii, self.diag_jj] = free_parameter[:, :self.size].exp() + self.delta
        return R

class Bias(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self):
        return self.bias

class Generative(nn.Module):
    def __init__(self, input_dim=784, dim_list = [20, 128, 256]):
        super().__init__()

        self.latent_dim_list = dim_list

        self.bias = Bias(dim_list[0])

        self.G_list = nn.ModuleList()
        for idx in range(len(dim_list)):
            module = nn.Sequential(nn.Identity())
            self.G_list.append(module)
        
        self.T_list = nn.ModuleList()
        for prev_dim, next_dim in zip(dim_list[:-1], dim_list[1:]):
            module = nn.Sequential(
                nn.ReLU(),
                nn.Linear(prev_dim, next_dim)
            )
            self.T_list.append(module)

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_list[-1], input_dim)
        )

    def forward(self, z_list):
        h = self.bias() + self.G_list[0](z_list[0])
        for G, T, z in zip(self.G_list[1:], self.T_list, z_list[1:]):
            h = T(h) + G(z)
        return torch.sigmoid(self.final(h))

    def sample(self, probs):
        return torch.distributions.Bernoulli(probs).sample()

    def sample_prior(self, batch_size, device = None):
        z_list = []
        for z_dim in self.latent_dim_list:
            z = torch.randn(batch_size, z_dim)
            if device is not None:
                z = z.to(device)
            z_list.append(z)
        return z_list

class RecognitionHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim, chol_factor_cls = None):
        super().__init__()

        self.chol_factor = chol_factor_cls(latent_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, self.chol_factor.free_parameter_size())

    def forward(self, h1):
        mu = self.fc21(h1)
        logvar_free = self.fc22(h1)
        R = self.chol_factor.parameterize(logvar_free)
        return mu, R
    
    def sample(self, mu, R):
        eps = torch.randn_like(mu)
        return mu + torch.einsum('ijk,ik->ij', R, eps) 

class RecognitionModelsShared(nn.Module):
    def __init__(self, input_dim=784, latent_dim_list = [20, 128, 256], hidden_dim = 64, chol_factor_cls = None):
        super().__init__()
 
        self.body = nn.Linear(input_dim, hidden_dim)
        self.node_list = nn.ModuleList()
        for latent_dim in latent_dim_list:
            node = RecognitionHead(hidden_dim, latent_dim, chol_factor_cls = chol_factor_cls)
            self.node_list.append(node)

    def forward(self, x):
        mu_list= []
        R_list = []
        for node in self.node_list:
            h = F.relu(self.body(x))
            mu, R = node(h)
            mu_list.append(mu)
            R_list.append(R)
        return mu_list, R_list

    def sample(self, mu_list, R_list):
        z_list = []
        for node, mu, R in zip(self.node_list, mu_list, R_list):
            z = node.sample(mu, R)
            z_list.append(z)
        return z_list
    
class Recognition(nn.Module):
    def __init__(self, input_dim=784, latent_dim = 20, hidden_dim = 400, chol_factor_cls = None):
        super().__init__()

        self.chol_factor = chol_factor_cls(latent_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, self.chol_factor.free_parameter_size())

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)

        logvar_free = self.fc22(h1)
        R = self.chol_factor.parameterize(logvar_free)
        return mu, R
    
    def sample(self, mu, R):
        eps = torch.randn_like(mu)
        return mu + torch.einsum('ijk,ik->ij', R, eps) 

class RecognitionModels(nn.Module):
    def __init__(self, input_dim=784, latent_dim_list = [20, 128, 256], hidden_dim=64, chol_factor_cls = None):
        super().__init__()

        self.node_list = nn.ModuleList()
        for latent_dim in latent_dim_list:
            node = Recognition(input_dim=input_dim, latent_dim = latent_dim, hidden_dim = hidden_dim, 
                chol_factor_cls = chol_factor_cls)
            self.node_list.append(node)

    def forward(self, x):
        mu_list= []
        R_list = []
        for node in self.node_list:
            mu, R = node(x)
            mu_list.append(mu)
            R_list.append(R)
        return mu_list, R_list

    def sample(self, mu_list, R_list):
        z_list = []
        for node, mu, R in zip(self.node_list, mu_list, R_list):
            z = node.sample(mu, R)
            z_list.append(z)
        return z_list

class DLGM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, factor_recog = 3,lr=1e-3, use_cuda=False):
        super().__init__()
        
        latent_dim_list = [latent_dim, hidden_dim, hidden_dim]
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # chol_factor_cls = cholesky_factor.CholeskyFactor 
        # chol_factor_cls = cholesky_factor.DiagonalFactor
        chol_factor_cls = RankOneFactor
        self.generative_model = Generative(input_dim=input_dim, dim_list=latent_dim_list).to(self.device)
        h = self.get_optimal_hidden_dim_recog(latent_dim_list, factor=factor_recog)        
        self.recognition_model = RecognitionModels(input_dim=input_dim, latent_dim_list=latent_dim_list, hidden_dim=h, chol_factor_cls = chol_factor_cls).to(self.device)
        self.optimizer = optim.Adam(chain(self.generative_model.parameters(), self.recognition_model.parameters()), lr=lr)

    def get_optimal_hidden_dim_recog(self, latent_dim_list, factor=3):
        n_gen = sum(p.numel() for p in self.generative_model.parameters())
        h = (factor*n_gen - 3*sum(latent_dim_list))//(len(latent_dim_list)*784+3*sum(latent_dim_list)+len(latent_dim_list))
        return h

    def set_optimizer(self, lr, decay=0.):
        self.optimizer = optim.Adam(chain(self.generative_model.parameters(), self.recognition_model.parameters()), lr=lr, weight_decay=decay)

    def get_nparameters(self):
        generative = sum(p.numel() for p in self.generative_model.parameters())
        recognition = sum(p.numel() for p in self.recognition_model.parameters())
        return {"#total":generative+recognition, "#generative":generative, "#recognition":recognition}

    def train(self, train_loader, epochs):
        self.generative_model.train()
        self.recognition_model.train()

        for epoch in range(1, epochs + 1):
            train_loss = 0
            num_steps = len(train_loader)
            with tqdm(total=num_steps) as pbar:
                for batch_idx, (data, _) in enumerate(train_loader):
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    
                    mu, R = self.recognition_model(data.view(-1, 28*28))
                    z = self.recognition_model.sample(mu, R)
                    recon_batch = self.generative_model(z)

                    loss = loss_function(recon_batch, data, mu, R)
                    
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()

                    pbar.set_description(f"Step {batch_idx+1}/{num_steps}, Loss={loss.item() / len(data):.4f}")
                    pbar.update(1)
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
    
    def test_default(self, test_loader, epoch):
        self.generative_model.eval()
        self.recognition_model.eval()

        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                #recon_batch, mu, logvar = model(data)
                mu, R = self.recognition_model(data.view(-1, 28*28))
                z = self.recognition_model.sample(mu, R)
                recon_batch = self.generative_model(z)

                test_loss += loss_function(recon_batch, data, mu, R).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                        recon_batch.view(recon_batch.shape[0], 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def generate_samples(self, num_samples, is_return_hidden=False):
        with torch.no_grad():
            sample = self.generative_model.sample_prior(num_samples, device = self.device)
            sample = self.generative_model(sample)
            if not is_return_hidden:
                sample = self.generative_model.sample(sample)
            return sample.view(-1, 28, 28)

    def get_fid(self, num_samples=5000, is_test=False):
        # check if MNIST_data/MNIST/test_img.npz exists
        if (not os.path.isfile("MNIST_data/MNIST/test_img.npz")) or (not os.path.isfile("MNIST_data/MNIST/val_img.npz")):
            print("MNIST_data/MNIST/test_img.npz or MNIST_data/MNIST/test_img.npz does not exist")
            print("Creating MNIST_data/MNIST/test_img.npz and MNIST_data/MNIST/val_img.npz")
            make_compressed_MNIST_files()
        images = self.generate_samples(num_samples, is_return_hidden=True)
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
        return float(str(result.stdout).split(" ")[2].split("\\")[0])

    def get_acc(self, loader):
        latent = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).type(torch.int).to(self.device)
        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)
            mu, R = self.recognition_model(data.view(-1, 28*28))
            latent, labels = torch.concatenate((latent, mu[0]), dim=0), torch.concatenate((labels, label), dim=0)
        dataset = TensorDataset(latent,labels)
        representations = DataLoader(dataset, batch_size=128, shuffle=True)
        classifier = MNIST_LinearClassifier(latent.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optm = optim.Adam(classifier.parameters(), lr=0.05) #, momentum=0.0

        # train classifier
        print("training classifier")
        EPOCHS = 50
        best_acc=0.
        for epoch_idx in range(EPOCHS):
            for data, label in representations:
                classifier.zero_grad()
                out = classifier(data)
                optm.zero_grad()
                loss = criterion(out, label)
                loss.backward(retain_graph=True)
                optm.step()
            acc = test(classifier,representations,print_acc=False)
            print("EPOCH ", epoch_idx, ": accuracy ", acc)
            if acc > best_acc:
                best_acc = acc
        print("Best accuracy: " + str(best_acc))
        return best_acc, classifier
    
    def get_mse_rec(self, loader):
        MSE, n_data = 0., 0
        for data, _ in loader:
            data = data.to(self.device)
            imgs = data.clone()
            data[:,:-round(data.shape[1]/2)] = 0.
            mu, R = self.recognition_model(data.view(-1, 28*28))
            x_hat = self.generative_model(mu)
            # MAP of Bernoulli
            x_hat = (x_hat>0.5).type_as(x_hat)
            MSE += ((x_hat[:,:-round(x_hat.shape[1]/2)] - imgs[:,:-round(imgs.shape[1]/2)])**2).mean(1).sum()
            n_data+= data.shape[0]
        return MSE/n_data

    def get_marginal_likelihood(self, dataloader, n_samples=5000):
        latent_samples = self.generate_samples(n_samples, is_return_hidden=True).cpu().reshape(1, -1, 784)
        latent_samples = torch.logit(latent_samples)
        latent_samples = torch.clamp(latent_samples, -20, 20)
        loss = nn.BCEWithLogitsLoss(reduction='none')
        # make appropriate dataloader
        dataloader_ml = DataLoader(dataloader.dataset, batch_size=int(100*6000/n_samples))
        losses = torch.tensor([])
        for data, _ in tqdm(dataloader_ml):
            batch_size = data.shape[0]
            data = data.unsqueeze(1) # reshape batch to [batch_size, 1, data_size(=latent_size)]
            data = data.repeat(1, n_samples,1)
            latent_samples_ = latent_samples.repeat(batch_size,1,1)
            l = loss(latent_samples_, data).view(batch_size, n_samples, latent_samples_.shape[-1]).sum(-1)
            losses = torch.concatenate((losses,l),dim=0)
        m_loss = losses.min(1).values
        p = torch.exp(-(losses-m_loss.unsqueeze(1))).mean(1)
        ml = (torch.log(p) - m_loss).mean()
        return ml
    
