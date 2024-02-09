import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
import tempfile, shutil
import os, subprocess


class Bias(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self):
        return self.bias


class MNIST_LinearClassifier(nn.Module):
    def __init__(self, rep_size):
        super().__init__()
        self.lin = nn.Linear(rep_size, 10)

    def forward(self, x):
        x = self.lin(x)
        return x

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


class BinaryMNIST(TensorDataset):
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        img = (img>0.5).type_as(img)
        return img, label


from torchvision import transforms, datasets
def get_mnist_data(config):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_dataset = BinaryMNIST(datasets.MNIST('../dataset', download=True, train=True, transform=transform))
    temp_dataset = datasets.MNIST('../dataset', download=True, train=False, transform=transform)
    val_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i for i in range(6000)]))
    test_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)]))    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size_val"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], shuffle=False)
    return train_loader, val_loader, test_loader



class metrics():
    def __init__(self, generative_model, recognition_model, use_cuda) -> None:
        self.generative_model = generative_model
        self.recognition_model = recognition_model
        self.device = torch.device("cuda" if use_cuda else "cpu")


    def generate_samples(self, num_samples, is_return_hidden=False):
        with torch.no_grad():
            sample = self.generative_model.sample_prior(num_samples, device = self.device)
            sample = self.generative_model(sample)
            if is_return_hidden:
                sample = self.generative_model.sample(sample)
            return sample.view(-1, 28, 28)


    def get_fid(self, num_samples=5000):
        images = self.generate_samples(num_samples)
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
        result = subprocess.run('python -m pytorch_fid val_img.npz '+ tf.name+"_", stdout=subprocess.PIPE, shell=True)
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
        pass

    def get_marginal_likelihood(self, dataloader, n_samples=5000):
        pass



