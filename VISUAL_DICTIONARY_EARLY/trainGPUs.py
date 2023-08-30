from datafolder import CustomDataset

import argparse
import os

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF
import random

from model import ConvVAE

""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """

## Arguments
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
# parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--batch_size_test', type=int, default=64, metavar='N')
parser.add_argument('--input_size', type=int, default=32, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='optimal_sigma_vae', metavar='N',
                    help='which model to use: mse_vae, gaussian_vae, sigma_vae, or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='optimal_sigma_vae_32', metavar='N')
args = parser.parse_args()

## Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

## Data Parallelism
if args.cuda and torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for Data Parallelism!")
    model = nn.DataParallel(ConvVAE(device, 3, args)).to(device)
else:
    model = ConvVAE(device, 3, args).to(device)

kwargs = {'num_workers': 48, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_dataset = CustomDataset(root_dir='/mayo_atlas/home/m288756/mayo_ai/data/download_and_patch_for_training/32_test/patches_32_test', transform=transform, trainTest="test")

# ## ---------------------------------------------------
train_dataset = CustomDataset(root_dir='/mayo_atlas/home/m288756/mayo_ai/data/download_and_patch_for_training/32_trainAll/patches_32_train', transform=transform, trainTest='train')



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, **kwargs)

## Logging
os.makedirs('vae_logs/{}'.format(args.log_dir), exist_ok=True)
summary_writer = SummaryWriter(log_dir='vae_logs/' + args.log_dir, purge_step=0)

## Build Model
optimizer = optim.Adam(model.parameters(), lr=0.0001)

## Define the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Run VAE
        recon_batch, mu, logvar = model(data)
        # Compute loss
        rec, kl = model.module.loss_function(recon_batch, data, mu, logvar)
        
        total_loss = rec + kl * 0.1
        # total_loss = rec + kl 
        total_loss.backward()
        train_loss += total_loss.item()
        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tlog_sigma: {:.6f}\tLR: {:.13f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec.item() / len(data),
                kl.item() / len(data),
                model.module.log_sigma if isinstance(model, nn.DataParallel) else model.log_sigma,
                scheduler.get_last_lr()[0]))
                
    train_loss /=  len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)
    summary_writer.add_scalar('train/log_sigma', model.module.log_sigma if isinstance(model, nn.DataParallel) else model.log_sigma, epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            rec, kl = model.module.loss_function(recon_batch, data, mu, logvar)
            test_loss += rec + kl * 0.1
            # test_loss += rec + kl
            if i == 2:
                n = min(data.size(0), 20)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size_test, -1, args.input_size, args.input_size)[:n]])
                save_image(comparison.cpu(), 'vae_logs/{}/reconstruction_{}.png'.format(args.log_dir, str(epoch)), nrow=n)
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    summary_writer.add_scalar('test/elbo', test_loss, epoch)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = model.module.sample(64).cpu() if isinstance(model, nn.DataParallel) else model.sample(64).cpu()
            save_image(sample.view(64, -1, args.input_size, args.input_size),
                    'vae_logs/{}/sample_{}.png'.format(args.log_dir, str(epoch)))
        summary_writer.file_writer.flush()
        scheduler.step()
        
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'vae_logs/{}/checkpoint_{}.pt'.format(args.log_dir, str(epoch)))
