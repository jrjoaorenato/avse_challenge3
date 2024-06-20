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
import soundfile as sf
import numpy as np


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    dataset = AVSEChallengeDataModule(data_root=args.data_root, batch_size=1, time_domain=True)
    model = DCUnet10CrossAttention(512)
    checkpoint_root = './checkpoint/'
    #print(summary(model, [(1, 257, 256, 2), (3, 64, 256, 256)]))
    #criterion = si_snr_loss
    #criterion = wsdr_fn_aud
    
    if args.mode == 'dev':
        test_dataset = dataset.dev_dataset
        output_root = args.save_root + 'dev/'
    elif args.mode == 'eval':
        test_dataset = dataset.test_dataset
        output_root = args.save_root + 'eval/'
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #load best model
    model.load_state_dict(torch.load(checkpoint_root + 'best_dev_loss_0.10106264596313876.pth'))
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            batch = test_dataset[i][0]
            noisy = batch['noisy_audio']
            bg_frames = batch['video_frames']
            filename = batch["scene"] + ".wav"
            
            # if the noisy file is greater than 2 seconds, we will split it into 5 seconds
            # chunks and process them separately
            pad_value = 0
            noisy_list = []
            if noisy.shape[0] > max_audio_length:
                for i in range(0, noisy.shape[0], max_audio_length):
                    #split from i to i+1
                    if noisy[i:].shape[0] >= max_audio_length:
                        noisy_list.append(torch.tensor(noisy[i:i+max_audio_length]).unsqueeze(0).to(device))
                    else:
                        #pad the last chunk
                        pad_value = max_audio_length-noisy[i:i+max_audio_length].shape[0]
                        temp_noisy = np.pad(noisy[i:i+max_audio_length], (0, max_audio_length-noisy[i:i+max_audio_length].shape[0]), 'constant')
                        noisy_list.append(torch.tensor(temp_noisy).unsqueeze(0).to(device))
                    #noisy_list.append(torch.tensor(noisy[i:i+max_audio_length]).unsqueeze(0).to(device))
            else:
                if noisy.shape[0] < max_audio_length:
                    pad_value = max_audio_length - noisy.shape[0]
                    noisy = np.pad(noisy, (0, max_audio_length - noisy.shape[0]), 'constant')
                noisy_list.append(torch.tensor(noisy).unsqueeze(0).to(device))
            
            if bg_frames.shape[1] >= max_video_length:
                #cut the video into one sample
                bg_frames = bg_frames[:, :max_video_length, :, :].unsqueeze(0).to(device)
            else:
                bg_frames = torch.cat([bg_frames, torch.zeros(bg_frames.shape[0], max_video_length - bg_frames.shape[1],
                                                               bg_frames.shape[2], bg_frames.shape[3])], dim=1)
                bg_frames = bg_frames.unsqueeze(0).to(device)
                
            est_clean_list = []
            for noisy in noisy_list:
                noisy_stft = torch.stft(noisy, win_length = window_size, n_fft = stft_size, hop_length=window_shift,\
                                    return_complex=True, window=torch.hann_window(window_size).to(device), normalized=True)
                noisy_stft = torch.view_as_real(noisy_stft).unsqueeze(1)
                
                noisy_stft = noisy_stft.to(device)
                bg_frames = bg_frames.to(device)
                
                if len(bg_frames.shape) == 4:
                    bg_frames = bg_frames.unsqueeze(0)
            
                est_clean = model(noisy_stft, bg_frames)
                est_clean = est_clean.squeeze().detach().cpu().numpy()
                # est_clean /= np.max(np.abs(est_clean))
                est_clean_list.append(est_clean)
            
            #concatenate all pieces and save
            #remove the padding from the last audio sample
            if pad_value > 0:
                if len(est_clean_list) > 1:
                    est_clean_list[-1] = est_clean_list[-1][:-pad_value]
                else:
                    est_clean_list[0] = est_clean_list[0][:-pad_value]
            complete_est_clean = np.concatenate(est_clean_list)
            complete_est_clean /= np.max(np.abs(complete_est_clean))
            sf.write(output_root+filename, complete_est_clean, samplerate=sampling_rate)
                
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./output/", help="Root directory to save enhanced audio")
    parser.add_argument("--data_root", type=str, default = './data', help="Root directory of dataset")
    parser.add_argument("--mode", type=str, default = 'eval', help="Type of data to extract")
    args = parser.parse_args()
    main(args)