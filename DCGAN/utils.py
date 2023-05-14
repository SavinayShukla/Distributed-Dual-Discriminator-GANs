import torch
import torchvision.transforms as transforms
import torch.optim as optim


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        arr.append(transforms.ToTensor()(pil_img))
    return torch.stack(arr)


def get_optimizer(parameters, lr=0.0002, betas=[0.5, 0.999]):
    return optim.Adam(parameters, lr=lr, betas=(betas[0], betas[1]))


def get_scheduler(optimizer, dataloader, epochs):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / (epochs*len(dataloader)))
