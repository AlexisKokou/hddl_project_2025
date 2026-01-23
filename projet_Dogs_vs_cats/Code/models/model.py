import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class CatDogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  
        features = self.fc[:-1](x) 
        out = self.fc[-1](features)
        if return_features:
            return features
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class BreedClassifier(nn.Module):
    def __init__(self, n_breed: int):
        super().__init__()
        self.n_breed = n_breed
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.MaxPool2d(2),  
            nn.Dropout(0.2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_breed)
        )
    def forward(self,x, return_features = False):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        features = self.fc[:-1](x)
        out = self.fc[-1](features)
        if return_features:
            return features
        else:
            return out 


class CatDogSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder1 = self.convblock(in_channels, 32)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.encoder2 = self.convblock(32, 64)
        self.encoder3 = self.convblock(64, 128)
        
        self.bottleneck = self.convblock(128, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.convblock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.convblock(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self.convblock(64, 32)
        
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def convblock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        
        b = self.bottleneck(self.maxpool(e3))
        
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        return self.out(d1)

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name="resnet34", num_classes=2):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head
        encoder_channels = self.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        cls_output = self.cls_head(features[-1])
        decoder_output = self.decoder(features)
        seg_output = self.segmentation_head(decoder_output)
        
        return cls_output, seg_output
