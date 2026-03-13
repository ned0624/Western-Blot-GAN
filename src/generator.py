import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(ConditionalGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, template1, template2):
        x = torch.cat((template1, template2), 1)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)        
        # 使用雙線性插值來匹配輸入尺寸
        x = torch.nn.functional.interpolate(x, size=(template1.size(2), template1.size(3)), mode='bilinear', align_corners=False)
        
        return x
