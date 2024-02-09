import shutil
import subprocess
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import os

from utils.model import bernoulli_fn, fe_fn


class BinaryMNIST(TensorDataset):
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        img = (img>0.5).type_as(img)
        return img, label
        

def get_mnist_data(config, binary=True):
    # Load MNIST data
    if config['loss_fn'] == fe_fn:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: torch.flatten(x))])
        train_dataset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
        temp_dataset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
        val_dataset = torch.utils.data.Subset(temp_dataset, [i for i in range(6000)])
        test_dataset = torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)])    
    elif (config['loss_fn'] == bernoulli_fn) or (config['loss_fn'] == "vae"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        temp_dataset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
        if binary:
            train_dataset = BinaryMNIST(datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform))
            val_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i for i in range(6000)]))
            test_dataset = BinaryMNIST(torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)]))    
        else:
            train_dataset = (datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform))
            val_dataset = (torch.utils.data.Subset(temp_dataset, [i for i in range(6000)]))
            test_dataset = (torch.utils.data.Subset(temp_dataset, [i+6000 for i in range(4000)]))    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size_val"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], shuffle=False)
    return train_loader, val_loader, test_loader


class GratingDataset(Dataset):
    # Your code here
    pass
    def __init__(self, num_samples, size=28, num_orientations=8):
        self.num_samples = num_samples
        self.size = size
        self.num_orientations = num_orientations

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        orientation_idx = np.random.randint(0, self.num_orientations)
        orientation_angle = (2 * np.pi / self.num_orientations) * orientation_idx

        grating_image = self.generate_grating_image(orientation_angle)

        # Convert to tensor and normalize
        grating_image = torch.tensor(grating_image, dtype=torch.float32)

        return torch.flatten(grating_image)

    def generate_grating_image(self, angle):
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        x_center, y_center = self.size // 2, self.size // 2
        x_rot = (x - x_center) * np.cos(angle) + (y - y_center) * np.sin(angle)
        y_rot = -(x - x_center) * np.sin(angle) + (y - y_center) * np.cos(angle)
        wavelength = 10  # Adjust the wavelength of the grating
        grating = np.sin(2 * np.pi * x_rot / wavelength)

        grating = grating/2 + 0.5

        return grating

class NoiseDataset(Dataset):
    def __init__(self, num_samples, size=28):
        self.num_samples = num_samples
        self.size = size

        self.imgs = (torch.rand(num_samples, size, size)>0.5).double()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.flatten(self.imgs[0])

class ChunkDataset(Dataset):
    def __init__(self, tensor, chunk_size):
        self.tensor = tensor
        self.chunk_size = chunk_size

    def __len__(self):
        return (self.tensor.size(0) - 1) // self.chunk_size + 1

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, self.tensor.size(0))
        return self.tensor[start_idx:end_idx]
    

def make_compressed_MNIST_files():
    config = {
        #
        "EPOCHS":10,
        "batch_size_train":256,
        "batch_size_val": 6000,
        "batch_size_test": 4000,
        #
        "input_size": 20,
        "hidden_size": 128,
        "hidden2_size": 128,
        "output_size": 784,
        "loss_fn": fe_fn ,
        "activation_fn": 'relu',
    }

    # Load MNIST data
    train_loader, val_loader, test_loader = get_mnist_data(config)

    # make compressed MNIST test file    
    data, label = list(test_loader)[0]
    images = data.view(-1,28,28)
    images = images/2 + 0.5
    # make test_img folder
    os.makedirs("test_img", exist_ok=True)
    print("Saving MNIST test images")
    for img_idx in tqdm(range(len(images))):
        save_image(images[img_idx], "test_img"+"\\"+str(img_idx)+".png")
    print("Compressing test images")
    subprocess.run('python -m pytorch_fid --save-stats test_img MNIST_data/MNIST/test_img.npz', stdout=subprocess.PIPE)
    # delete test_img and all contents 
    shutil.rmtree("test_img")

    # make compressed MNIST validation file
    data, label = list(val_loader)[0]
    images = data.view(-1,28,28)
    images = images/2 + 0.5
    # make val_img folder
    os.makedirs("val_img", exist_ok=True)
    print("Saving MNIST validation images")
    for img_idx in tqdm(range(len(images))):
        save_image(images[img_idx], "val_img"+"\\"+str(img_idx)+".png")
    print("Compressing validation images")
    subprocess.run('python -m pytorch_fid --save-stats val_img MNIST_data/MNIST/val_img.npz', stdout=subprocess.PIPE)
    # delete val_img and its contents
    shutil.rmtree("val_img")
