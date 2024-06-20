import math
import numpy as np
import torch
from torch import nn
from config import *

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output
    
class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output
    
class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted
    
class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

class CrossAttention(nn.Module):
    #https://www.kaggle.com/code/aisuko/coding-cross-attention
    def __init__(self, d_in = 512, d_out_kq = 8640, d_out_v = 8640):
        super().__init__()
        self.d_out_kq=d_out_kq
        self.W_query= nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key  = nn.Parameter(torch.rand(d_out_kq, d_out_kq))
        self.W_value= nn.Parameter(torch.rand(d_out_kq, d_out_v))
        
    def forward(self, x1, x2):
        B, C, H, W, P = x2.shape
        x2 = x2.view(B, C * H * W * P)
        
        queries_1=x1.matmul(self.W_query)
        keys_2=x2.matmul(self.W_key)
        values_2=x2.matmul(self.W_value)
        
        attn_scores=queries_1.matmul(keys_2.T)
        attn_weights=torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )
        context_vec=attn_weights.matmul(values_2)
        
        context_vec = context_vec.view(B, C, H, W, P)
        return context_vec
        
           
class DCUnet10(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """
    def __init__(self):
        super().__init__()
        # for istft
        self.cross_attention = CrossAttention(512)
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=90, output_padding=(0,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, context, is_istft=True):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)        
        d3 = self.downsample3(d2)        
        d4 = self.downsample4(d3)
        
        d4 = self.cross_attention(context, d4)
        
        # upsampling/decoding 
        u0 = self.upsample0(d4)
        # pad u0 to match the shape of d3
        u0 = torch.nn.functional.pad(u0, (0, 0, 0, 0, 0, 1, 0, 0))
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)
        
        u1 = self.upsample1(c0)
        # pad u1 shape [2, 90, 27, 29, 2] to [2, 90, 28, 30, 2]
        u1 = torch.nn.functional.pad(u1, (0, 0, 0, 1, 0, 1, 0, 0))
        c1 = torch.cat((u1, d2), dim=1)
        
        u2 = self.upsample2(c1)
        #pad u2 to match the shape of d1
        u2 = torch.nn.functional.pad(u2, (0, 0, 0, 0, 0, 1, 0, 0))
        c2 = torch.cat((u2, d1), dim=1)
        
        u3 = self.upsample3(c2)
        #ád u3 to match the shape of d0
        u3 = torch.nn.functional.pad(u3, (0, 0, 0, 1, 0, 1, 0, 0))
        c3 = torch.cat((u3, d0), dim=1)
        
        u4 = self.upsample4(c3)
        
        # u4 - the mask
        output = u4 * x
        if is_istft:
            with torch.enable_grad():
                output = torch.squeeze(output, 1)
                output = torch.view_as_complex(output).contiguous()
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                else:
                    device = torch.device('cpu')
                #requires gradient true in output
                output = torch.istft(output, win_length = window_size, n_fft = stft_size, hop_length=window_shift, return_complex=False, window=torch.hann_window(window_size).to(device))
            
        return output
    
if __name__ == '__main__':
    model = DCUnet10()
    model = model.to(torch.device('cuda:0'))
    context = torch.randn(2, 512).to(torch.device('cuda:0'))
    x = torch.randn(2,40800)
    x_feat = torch.stft(x, win_length = window_size, n_fft = stft_size, hop_length=window_shift, return_complex=True, window=torch.hann_window(window_size), normalized=True)
    x_feat = torch.view_as_real(x_feat)
    x_feat = x_feat.unsqueeze(1)
    print(x_feat.shape)
    x_feat = x_feat.to(torch.device('cuda:0'))
    y = model(x_feat, context)
    print(y.shape)
    print(y)