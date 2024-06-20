import math
import numpy as np
import torch
import transformers
from torch import nn
import timm
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from pytorch_tcn import TCN
import torchsummary
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import s3d, S3D_Weights


class VideoEncoder(nn.Module):
    def __init__(self, embedding_size = 512):
        super(VideoEncoder, self).__init__()
        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        #add the projection layer
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv3d(1024, embedding_size, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )
                
    def forward(self, x):
        #x is a tensor of shape (batch_size, num_frames, 3, 224, 224)
        #extract features from each frame
        x = self.cnn(x)
        return x

class ViMEncoder(nn.Module):
    def __init__(self, num_frames):
        super(ViMEncoder, self).__init__()
        #use vision mamba to encode all the frames in a video and pass them to a TCN
        #vision mamba is the vim_small_patch16_224 from timm
        
        #self.vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
        
        #change the input size of the model and the processor to img_size
        #self.vit.config.image_size = (img_size, img_size)
        #self.cnn = timm.create_model('resnet18', pretrained=True)
        self.cnn = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True)
        #change classifier layer to 512
        self.cnn.classifier = nn.Linear(1280, 512)
        self.num_frames = num_frames
        #remove the last layer of the resnet model
        # self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
    
        #use a TCN to relate the features extracted from the frames
        self.frontend3D = nn.Sequential(
            nn.Conv3d(self.num_frames, self.num_frames, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.num_frames),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        )
        self.tcn = TCN(
            num_inputs=self.num_frames,
            num_channels=[self.num_frames, self.num_frames, self.num_frames, self.num_frames],
            kernel_size=7,
            dilations = [1, 2, 4, 8],
            dropout = 0.2,
            use_skip_connections=True,
        )
        #project the features to the latent space
        #self.projector = nn.Linear(512*num_frames, 2048)
                
    def forward(self, x):
        #x is a tensor of shape (batch_size, num_frames, 3, 224, 224)
        #extract features from each frame
        x = self.frontend3D(x)
        B, T, C, H, W = x.shape
        if B is None:
            B = 1
        x = x.reshape(-1, C, H, W)
        #x = x.permute(0, 2, 1, 3, 4)
        #x = x.reshape(-1, 3, 224, 224)
        x = self.cnn(x)
        x = x.reshape(-1, self.num_frames, 512)
        x = self.tcn(x)
        #flatten the features
        #x = x.reshape(B, -1)
        #x = self.projector(x)
        return x
    
if __name__ == '__main__':
    preprocess = S3D_Weights.KINETICS400_V1.transforms()
    model = VideoEncoder()
    #print(model)
    x = torch.randn(1, 127, 3, 88, 88)
    x = preprocess(x)
    print(x.shape)
    x = x.to(torch.device('cuda:0'))
    model = model.to(torch.device('cuda:0'))
    torchsummary.summary(model, (3, 127, 256, 256))
    print(model(x).shape)
    # model = ViMEncoder(127)
    # model = model.to(torch.device('cuda:0'))
    # torchsummary.summary(model, (127, 3, 224, 224))
    # x = torch.randn(1, 127, 3, 224, 224)
    # x = x.to(torch.device('cuda:0'))
    # print(model(x).shape)
        
        