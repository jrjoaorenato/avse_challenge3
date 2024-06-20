import numpy as np
import torch
from torch import nn
from config import *
import torchsummary
from model_utils.visual import VideoEncoder
from model_utils.unet import DCUnet10
from torchvision.models.video import s3d, S3D_Weights
from thop import profile
from tqdm import tqdm

class DCUnet10CrossAttention(nn.Module):
    def __init__(self, av_embedding = 512):
        super(DCUnet10CrossAttention, self).__init__()
        self.video_encoder = VideoEncoder(embedding_size = av_embedding)
        self.unet = DCUnet10()  
        
    def forward(self, audio, video):        
        video_context = self.video_encoder(video)          
        pred_audio = self.unet(audio, video_context)
        return pred_audio
        
    
    
if __name__ == '__main__':
    model = DCUnet10CrossAttention(512)
    preprocess = S3D_Weights.KINETICS400_V1.transforms()
    #torchsummary.summary(model, [(1, 257, 256, 2), (3, 64, 256, 256)])
    model = model.to(torch.device('cuda:0'))
    audio = torch.randn(1, 40800)
    x_feat = torch.stft(audio, win_length = window_size, n_fft = stft_size, hop_length=window_shift, return_complex=True, window=torch.hann_window(window_size), normalized=True)
    x_feat = torch.view_as_real(x_feat)
    x_feat = x_feat.unsqueeze(1)
    x_feat = x_feat.to(torch.device('cuda:0'))
    video = torch.randn(1, 64, 3, 256, 256)
    video = preprocess(video)
    #fix frame size to 128
    video = video.to(torch.device('cuda:0'))
    #with torch.no_grad():
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(x_feat, video)
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(x_feat, video)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn, "+-", std_syn, "ms")
    

    
    
    # y = model(x_feat, video)
    # print(y.shape)
    # print(audio, y)