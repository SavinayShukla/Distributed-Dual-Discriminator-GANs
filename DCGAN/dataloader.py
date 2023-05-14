
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

# Transformations based on the datasets below.
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_bw = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Returns the dataloader
def get_loader(type="cifar10", batch_size = 128, workers=2):

    if type == 'cifar10':
        dataset = datasets.CIFAR10(root='data/cifar10/', train=True, download=True, transform=transform)

    if type == 'stl':
        dataset = datasets.STL10(root='data/stl/', split='train', download=True, transform=transform)

    if type == 'svhn':
        dataset = datasets.SVHN(root='data/svhn/', split='train', download=True, transform=transform)

    if type == 'mnist':
        dataset = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform_bw)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader