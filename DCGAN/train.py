import random
import torch
import torch.nn as nn
from utils import get_optimizer, get_scheduler, interpolate
from dataloader import get_loader
import time
import torch.backends.cudnn as cudnn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torchvision.utils as vutils
from torcheval.metrics import Throughput
from models import get_discriminator, get_generator
import os
import json
import numpy as np
import configparser


def run(epochs, generator, discriminator, loader, device, hparams):
    batch_size = hparams["batch_size"]
    nz = hparams["nz"]
    lr = hparams["lr"]
    betas = hparams["betas"]

    logs = dict()

    real_batch = next(iter(loader))
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    inception = InceptionScore(normalize=True)
    metric_with_loader = Throughput()
    metric_without_loader = Throughput()
    criterion = criterion = nn.BCELoss()

    optimizerG = get_optimizer(generator.parameters(), lr = lr, betas = betas)
    optimizerD = get_optimizer(discriminator.parameters(), lr = lr, betas = betas)

    schedularG = get_scheduler(optimizer = optimizerG, dataloader=loader, epochs=epochs)
    schedularD = get_scheduler(optimizer = optimizerD, dataloader=loader, epochs=epochs)

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    iterations = 0
    avg_dataloading_time = 0.0
    calc_time = 0.00
    thput_with_loader_sum = 0.00
    thput_without_loader_sum = 0.00
    total_training_time = 0.00

    generator_loss = []
    discriminator_loss = []
    IS_scores = []
    FID_scores = []
    peak_memory = []
    device_utilization = []
    print("\n##########################################################################")
    print("Starting Training....\n")
    for epoch in range(epochs):
        g_loss = 0.0
        d_loss = 0.0
        # For each batch in the dataloader
        generator.train()
        discriminator.train()
        calculation_time = 0.00
        dataloadingtime_start = time.monotonic ()
        utilization = 0.00
        cumulative_memory_usage = 0.00
        total = 0
        for i, data in enumerate(loader, 0):
            calc_start = time.monotonic()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            # output = netD(fake.detach()).view(-1)
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            schedularG.step()
            schedularD.step()
            calc_end = time.monotonic()
            calculation_time += calc_end - calc_start
            # Save Losses for plotting later
            g_loss += errG.item()
            d_loss += errD.item()
            utilization += torch.cuda.utilization(0)
            max_mem_allocated = torch.cuda.max_memory_allocated()
            cumulative_memory_usage += (max_mem_allocated/1024**2)
            total += b_size
            iterations += 1

        cumulative_memory_usage = cumulative_memory_usage/len(loader)
        peak_memory.append(cumulative_memory_usage)
        dataloadingtime_end = time.monotonic()
        training_time = (dataloadingtime_end - dataloadingtime_start)

        total_training_time += training_time
        dataloadingtime = (dataloadingtime_end - dataloadingtime_start) - calculation_time
        avg_dataloading_time += dataloadingtime
        calc_time += calculation_time

        metric_with_loader.update(total, training_time)
        metric_without_loader.update(total, calculation_time)
        thput_with_loader = metric_with_loader.compute()
        thput_without_loader = metric_without_loader.compute()
        thput_with_loader_sum += thput_with_loader.item()
        thput_without_loader_sum += thput_without_loader.item()

        avg_g_loss = g_loss / total
        avg_d_loss = d_loss / total
        avg_utilization = utilization / len(loader)
        generator_loss.append(avg_g_loss)
        discriminator_loss.append(avg_d_loss)
        device_utilization.append(avg_utilization)

        with torch.no_grad():
            fake_batch = generator(fixed_noise).detach().cpu()
            if hparams["type"] == "mnist":
                vutils.save_image(fake_batch.data, '%s/fake_samples_epoch_%s.png' % ("results/"+hparams["type"], str(epoch)))
            else:
                vutils.save_image(fake_batch.data, '%s/fake_samples_epoch_%s.png' % ("results/"+hparams["type"], str(epoch)), normalize=True)
            fake = interpolate(fake_batch)
            inception.update(fake)
            fid.update(real_batch[0], real=True)
            fid.update(fake, real=False)
            FID = fid.compute()
            IS,_ = inception.compute()
            print(f"Epoch: {epoch}/{iterations}, G_Loss: {avg_g_loss}, D_Loss: {avg_d_loss}\nIS: {IS.item()}, FID: {FID.item()}\nDataLoading Time: {dataloadingtime}, Calculation Time: {calculation_time}, Training Time: {training_time}\nUtilization: {avg_utilization}, Throughput (with Loading): {thput_with_loader}, Throughput (without Loading): {thput_without_loader}, Peak Memory: {cumulative_memory_usage:.2f} MB")
            print("**********************************************************")
            IS_scores.append(IS.item())
            FID_scores.append(FID.item())
            fid.reset()

    peak_mem_avg = np.mean(peak_memory)
    avg_training_time = total_training_time /epochs
    avg_thrpt_with_loader = thput_with_loader_sum/epochs
    avg_thrpt_without_loader = thput_without_loader_sum/epochs
    avg_dataloading_time = avg_dataloading_time/epochs
    avg_utilization = np.mean(device_utilization)
    avg_calc = calc_time / epochs

    torch.save(generator.state_dict(), './saved/generator.pth')
    torch.save(discriminator.state_dict(), './saved/discriminator.pth')

    logs["g_loss"] = generator_loss
    logs["d_loss"] = discriminator_loss
    logs["IS"] = IS_scores
    logs["FID"] = FID_scores
    logs["avg_dl_time"] = avg_dataloading_time
    logs["utilization"] = avg_utilization
    logs["avg_thrpt_loader"] = avg_thrpt_with_loader
    logs["avg_thrpt"] = avg_thrpt_without_loader
    logs["avg_training_time"] = avg_training_time
    logs["avg_calc"] = avg_calc
    logs["peak_mem_avg"] = peak_mem_avg
    logs["iterations"] = iterations


    return logs

if __name__ == "__main__":

    cudnn.benchmark = True
    random.seed(100)
    torch.manual_seed(100)
    torch.cuda.empty_cache()

    config = configparser.ConfigParser()
    config.read(os.path.dirname(os.path.abspath(__file__))+"/config.ini")

    hparams = dict()

    hparams["type"] = str(config["parameters"]['type'])
    hparams["batch_size"] = config.getint("parameters", "batch_size")
    hparams["nz"] = config.getint("parameters", "nz")
    hparams["ngf"] = config.getint("parameters", "ngf")
    hparams["ndf"] = config.getint("parameters", "ndf")
    hparams["nc"] = config.getint("parameters", "nc")
    hparams["lr"] = config.getfloat("parameters", "lr")
    hparams["betas"] = [config.getfloat("parameters", "beta1"), config.getfloat("parameters", "beta2")]
    hparams["workers"] = config.getint("parameters", "workers")
    hparams["epochs"] = config.getint("parameters", "epochs")

    print("##########################################################################")
    print("Random Seed: ", 100)
    print(f"Dataset : {hparams['type']}")
    print(f"Batch Size : {hparams['batch_size']}")
    print(f"Latent Noise Dimension : {hparams['nz']}")
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

    if not os.path.exists('results/'+hparams["type"]):
        print("Result sub directory...")
        os.mkdir('results/'+hparams["type"])
  
    print(f"Loading Dataset: {hparams['type']}")

    dataloader = get_loader(batch_size=hparams["batch_size"], type=hparams["type"], workers=hparams["workers"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = get_generator(hparams["nz"], nc=hparams["nc"], ngf=hparams["ngf"]).to(device)
    discriminator = get_discriminator(hparams["nz"], nc=hparams["nc"], ndf=hparams["ndf"]).to(device)
    
    logs = run(hparams["epochs"], generator, discriminator, dataloader, device, hparams)
    is_file = 'logs/log_'+hparams['type']+'_'+str(time.time())+'.json'          #use the file extension .json
    with open(is_file, 'w') as file_object:  #open the file in write mode
        json.dump(logs, file_object) 

