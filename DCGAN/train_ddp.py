import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import datasets
import time
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torcheval.metrics import Throughput
from torch.utils.data.distributed import DistributedSampler
import json

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12351'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        super(Generator, self).__init__()
        self.main = nn.Sequential(

            # Z : nz x 1 x 1
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf * 8) x 2 x 2
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # ngf x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.nz, 1, 1))
        return output

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc):
        self.nz = nz
        self.ndf = ndf
        self.nc = nc
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
def runner(rank, world_size, hprams):
    batch_size = hprams["batch_size"]
    nz = hprams["nz"]
    lr = hprams["lr"]
    betas = hprams["betas"]
    epochs = hprams["epochs"]
    workers = hprams["workers"]
    setup(rank, world_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device and Rank: ", device, rank)
    torch.cuda.set_device(rank)
    
    transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    dataset = datasets.CIFAR10(root='data/cifar10/', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size, shuffle=False, sampler=sampler,
        num_workers = workers)
    metric_loader = Throughput()
    metric = Throughput()
  
    generator_ = Generator(nz=hprams["nz"], ngf=hprams["ngf"], nc=hprams["nc"])
    discriminator_ = Discriminator(nz=hprams["nz"], ndf=hprams["ndf"], nc=hprams["nc"])
    
    generator_.apply(weights_init)
    generator_.to(rank)

    discriminator_.apply(weights_init)
    discriminator_.to(rank)
    
    generator_ = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator_)
    discriminator_ = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator_)

    generator = DDP(generator_, device_ids=[rank])
    discriminator = DDP(discriminator_, device_ids=[rank])

    criterion = nn.BCELoss().cuda()
    fixed_noise = torch.randn(batch_size, nz, 1, 1).cuda()

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.99))
    optimizerG = optim.Adam(generator.parameters(), lr=lr,betas=(0.5, 0.99))
    
    schedularG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lambda step: 1 - step / (epochs*len(loader)))
    schedularD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lambda step: 1 - step / (epochs*len(loader)))

    generator_loss, discriminator_loss = [], []
    peak_memory = []
    avg_dataloading_time = 0.0
    calc_time = 0.00
    thput_sum = 0.00
    total_training_time = 0.00
    dist_training_start = time.monotonic()

    logs = dict()
    
    print("Starting Distrubuted Training")
    for epoch in range(epochs):
        loader.sampler.set_epoch(epoch)
        g_loss = 0.0
        d_loss = 0.0
        D_x = 0.0
        D_G_z1 = 0.0
        D_G_z2 = 0.0
        calculation_time = 0.00
        dataloadingtime_start = time.monotonic ()
        utilization = 0.00
        cumulative_memory_usage = 0.00
        generator.train()
        discriminator.train()
        total = 0
        for i, data in enumerate(loader, 0):
            calc_start = time.monotonic()
            discriminator.zero_grad()
            real_cpu = data[0].cuda()
            data_length = len(real_cpu)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
            # Discriminator start
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x += output.mean().item()

            noise = torch.randn(b_size, 100, 1, 1).cuda()
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 += output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            #Discriminator end

            #Generator start
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 += output.mean().item()
            optimizerG.step()
            schedularG.step()
            schedularD.step()
            
            calc_end = time.monotonic()
            calculation_time += calc_end - calc_start
            total += b_size
            #Generator end

            g_loss += errG.item()
            d_loss += errD.item()
            utilization += torch.cuda.utilization([rank])
            max_mem_allocated = torch.cuda.max_memory_allocated()
            cumulative_memory_usage += (max_mem_allocated/1024**2) / world_size
        
        peak_memory.append(cumulative_memory_usage)
        dataloadingtime_end = time.monotonic()
        training_time = (dataloadingtime_end - dataloadingtime_start)
        total_training_time += training_time
        
        dataloadingtime = (dataloadingtime_end - dataloadingtime_start) - calculation_time
        avg_dataloading_time += dataloadingtime
        calc_time += calculation_time
        metric_loader.update(total, training_time)
        metric.update(total, calculation_time)
        thput_loader = metric_loader.compute() * world_size
        thput = metric.compute() * world_size
        thput_sum += thput
        avg_g_loss = g_loss / total
        avg_d_loss = d_loss / total
        
        avg_utilization = utilization / len(loader)
        generator_loss.append(avg_g_loss)
        discriminator_loss.append(avg_d_loss)
        if(rank==0):
            print(f"Epoch: {epoch}, G_Loss: {avg_g_loss}, D_Loss: {avg_d_loss}\nUtilization: {avg_utilization}, Throughput(with Loaders): {thput_loader}, Throughput: {thput}, Epoch Time: {training_time}, Peak Memory: {cumulative_memory_usage:.2f} MB")
            print("**********************************************************")

    if(rank==0):     
        print('\nDistributed Training Completed...')
    dist_training_end = time.monotonic()

    if(rank==0):
        peak_mem_avg = np.mean(peak_memory)
        avg_training_time = total_training_time /epochs
        avg_thrpt = thput_sum/epochs
        avg_dataloading_time = avg_dataloading_time/epochs
        avg_calc = calc_time / epochs
        print(f"Average Training Time per device: {avg_training_time} sec")
        print(f"Total Distributed Training Time: {dist_training_end - dist_training_start} sec")
        print(f"Average Throughput per device: {avg_thrpt}")
        print(f"Average Data Loading Time per device: {avg_dataloading_time} sec")
        print(f"Average Calculation Time: {avg_calc} sec")
        print(f"Peak Memory Usage: {peak_mem_avg / world_size}")

        logs["g_loss"] = generator_loss
        logs["d_loss"] = discriminator_loss
        logs["avg_dl_time"] = avg_dataloading_time
        logs["utilization"] = avg_utilization
        logs["avg_thrpt"] = avg_thrpt
        logs["avg_training_time"] = avg_training_time
        logs["avg_calc"] = avg_calc
        logs["peak_mem_avg"] = peak_mem_avg / world_size

        is_file = 'logs/log_'+str(time.time())+'.json'
        with open(is_file, 'w') as file_object:  
            json.dump(logs, file_object)

    cleanup()

if __name__ == "__main__":
    
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print("Changes observed!")
    hparams = dict()
    hparams["type"] = "cifar10"
    hparams["batch_size"] = 128
    hparams["nz"] = 100
    hparams["ngf"] = 64
    hparams["ndf"] = 64
    hparams["nc"] = 3
    hparams["lr"] = 0.0002
    hparams["betas"] = [0.5, 0.999]
    hparams["checkpoint"] = 200
    hparams["workers"] = 1
    hparams["epochs"] = 30

    print(f"Dataset : {hparams['type']}")
    print(f"Batch Size : {hparams['batch_size']}")
    print(f"Latent Noise Dimenstion : {hparams['nz']}")
    print(f"Generator Filters : {hparams['ngf']}")
    print(f"Discriminator Filters : {hparams['ndf']}")
    print(f"Channels : {hparams['nc']}")
    print(f"Learning Rate : {hparams['lr']}")
    print(f"Workers : {hparams['workers']}")
    print(f"Epochs : {hparams['epochs']}")

    if not os.path.exists("results"):
        print("Making Results directory...")
        os.mkdir("results")

    if not os.path.exists("logs"):
        print("Making Logs/Stats directory...")
        os.mkdir("logs")

    if not os.path.exists("results/"+hparams["type"]):
        print("Result sub directory...")
        os.mkdir("results/"+hparams["type"])

    mp.spawn(runner,
             args=(world_size,hparams),
             nprocs=world_size,
             join=True)