import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy
import math
from config import *
import argparse

from model import DCUnet10CrossAttention
from dataset import AVSEChallengeDataModule
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
#model summary
from torchsummary import summary
from loss import *


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    dataset = AVSEChallengeDataModule(data_root=args.data_root, batch_size=args.batch_size, time_domain=True)
    model = DCUnet10CrossAttention(512)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,\
                                                           min_lr=10**(-10), cooldown=1, verbose=True)
    checkpoint_root = './checkpoint/'
    writer = SummaryWriter(log_dir='./logs')
    print(summary(model, [(1, 257, 256, 2), (3, 64, 256, 256)]))
    criterion = si_snr_loss
    #criterion = wsdr_fn_aud
    
    train_loader = dataset.train_dataloader()
    dev_loader = dataset.val_dataloader()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    for i in range(args.epochs):
        avg_epoch_train_loss = 0
        avg_epoch_dev_loss = 0
        BEST_DEV = 100000
        BEST_TRAIN = 100000
        n_steps = -1
        model.train()
        #create and updatable progress bar
        for x, clean in (pbar := tqdm(train_loader, desc=f'epoch {i} loss 0')):
            if n_steps == -1:
                n_steps = 0
            noisy = x['noisy_audio'].to(device)
            clean = clean.to(device)
            bg_frames = x['video_frames'].to(device)
            optimizer.zero_grad()
            
            noisy_stft = torch.stft(noisy, win_length = window_size, n_fft = stft_size, hop_length=window_shift,\
                                    return_complex=True, window=torch.hann_window(window_size).to(device), normalized=True)
            noisy_stft = torch.view_as_real(noisy_stft).unsqueeze(1)
            
            est_clean = model(noisy_stft, bg_frames)
            loss = criterion(est_clean, clean)
            #loss = criterion(noisy, est_clean, clean)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, i)
            avg_epoch_train_loss += loss.item()
            n_steps += 1
            #update description
            pbar.set_description(f'epoch {i} loss {loss.item()}')
        #print current average loss
        avg_epoch_train_loss /= n_steps
        print(f'Epoch {i} train loss: {avg_epoch_train_loss}')
        #save checkpoint
        if avg_epoch_train_loss < BEST_TRAIN:
            BEST_TRAIN = avg_epoch_train_loss
            torch.save(model.state_dict(), checkpoint_root + f'best_train_loss_{BEST_TRAIN}.pth')
        avg_epoch_train_loss = 0
        n_steps = 0
        model.eval()
        with torch.no_grad():
            for x, clean in tqdm(dev_loader):
                noisy = x['noisy_audio'].to(device)
                clean = clean.to(device)
                bg_frames = x['video_frames'].to(device)
                optimizer.zero_grad()
                
                noisy_stft = torch.stft(noisy, win_length = window_size, n_fft = stft_size, hop_length=window_shift,\
                                        return_complex=True, window=torch.hann_window(window_size).to(device), normalized=True)
                noisy_stft = torch.view_as_real(noisy_stft).unsqueeze(1)
                
                est_clean = model(noisy_stft, bg_frames)
                val_loss = criterion(est_clean, clean)
                #val_loss = criterion(noisy, est_clean, clean)
                writer.add_scalar('Loss/dev', val_loss, i)
                avg_epoch_dev_loss += val_loss.item()
                n_steps += 1
            avg_epoch_dev_loss /= n_steps
            print(f'Epoch {i} dev loss: {avg_epoch_dev_loss}')
            if avg_epoch_dev_loss < BEST_DEV:
                BEST_DEV = avg_epoch_dev_loss
                torch.save(model.state_dict(), checkpoint_root + f'best_dev_loss_{BEST_DEV}.pth')
            torch.save(model.state_dict(), checkpoint_root + f'epoch_{i}.pth')
            writer.flush()
            avg_epoch_dev_loss = 0
            n_steps = 0
        reduce_lr.step(val_loss)
    torch.save(model.state_dict(), checkpoint_root + 'final.pth')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
    
