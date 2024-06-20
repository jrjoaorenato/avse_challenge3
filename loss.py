import torch
import os
from config import *

def l2_norm(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)

def si_snr_loss(x, y, eps = 1e-8):
    s1_s2_norm = l2_norm(x, y)
    s2_s2_norm = l2_norm(y, y)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * y
    e_nosie = x - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return -1 * torch.mean(snr)

def wsdr_fn_aud(noisy, clean_pred, clean_true, eps=1e-8):
    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = noisy - clean_true
    z_pred = noisy - clean_pred

    a = torch.sum(clean_true**2, dim=1) / (torch.sum(clean_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(clean_true, clean_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    if torch.cuda.is_available():
        device = torch.device('cuda')
    y_true_ = torch.squeeze(y_true_, 1)
    x_ = torch.squeeze(x_, 1)
    y_true_ = torch.view_as_complex(y_true_)
    x_ = torch.view_as_complex(x_)
    x = torch.istft(x_, win_length = window_size, n_fft = stft_size, hop_length=window_shift,\
                         return_complex=False, window=torch.hann_window(window_size).to(device))
    y_true = torch.istft(y_true_, win_length = window_size, n_fft = stft_size, hop_length=window_shift,\
                         return_complex=False, window=torch.hann_window(window_size).to(device))
    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

if __name__ == '__main__':
    #x = torch.randn(4, 1, 257, 256, 2).to(torch.device('cuda'))
    #y = torch.randn(4, 1, 257, 256, 2).to(torch.device('cuda'))
    x = torch.randn(4, 40800).to(torch.device('cuda'))
    y = torch.randn(4, 40800).to(torch.device('cuda'))
    z = torch.randn(4, 40800).to(torch.device('cuda'))
    #print(si_snr_loss(x, y))
    print(wsdr_fn_aud(x, z, y))