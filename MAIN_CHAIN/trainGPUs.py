from custom_dataset import CustomDataset

import argparse
import os
import time 

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF

from model_vae_8_dim import Autoencoder

""" This script is an example of VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

# I always save the checkpoint after the 15 epoch because it has been providing good results

## Arguments
parser = argparse.ArgumentParser(description='VAE Training Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
# parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--batch_size_test', type=int, default=120, metavar='N')
parser.add_argument('--input_size', type=int, default=180, metavar='N')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='mse_vae', metavar='N',
                    help='which model to use: mse_vae, gaussian_vae, sigma_vae, or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='mse_vae', metavar='N')
args = parser.parse_args()


## Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:2" if args.cuda else "cpu")

# ## Data Parallelism
# if args.cuda and torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs for Data Parallelism!")
#     model = nn.DataParallel(Autoencoder(args.input_size)).to(device)
# # else:
model = Autoencoder(args.input_size).to(device)

kwargs = {'num_workers': 48, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



# ## ---------------------------------------------------
train_dataset = CustomDataset(data_path='/mayo_atlas/home/m296984/RESULTS_40x/Liver/test_180x180_patches', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

## Logging
os.makedirs('vae_logs/{}'.format(args.log_dir), exist_ok=True)
summary_writer = SummaryWriter(log_dir='vae_logs/' + args.log_dir, purge_step=0)

## Build Model
optimizer = optim.Adam(model.parameters(), lr=0.0001)

## Define the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Run VAE
        recon_batch, mu, logvar = model(data)
        # Compute loss
        rec, kl = model.loss_function(recon_batch, data, mu, logvar)
        
        total_loss = rec + kl * 0.1
        # total_loss = rec + kl 
        total_loss.backward()
        train_loss += total_loss.item()
        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tLR: {:.13f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec.item() / len(data),
                kl.item() / len(data),
                scheduler.get_last_lr()[0]))
                
    train_loss /=  len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)


if __name__ == "__main__":
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        summary_writer.file_writer.flush()
        scheduler.step()
        # Where you want to save all the results (in like a vae log)
        os.makedirs(f'/mayo_atlas/home/m296984/RESULTS_40x/Liver/vae_logs/{args.log_dir}', exist_ok=True)
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                '/mayo_atlas/home/m296984/RESULTS_40x/Liver/vae_logs/{}/checkpoint_{}.pt'.format(args.log_dir, str(epoch)))
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training model time ({args.epochs}): {execution_time:.2f} seconds")

