from .model_parts import *
from torchvision import models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes, dropout = 0.2, pretrained = True, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pretrained = pretrained
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.sigmoid(logits)
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet34(nn.Module):
    def __init__(self, n_classes, dropout = 0.2):
        super(UNet34, self).__init__()
        
        # Use pretrained ResNet-34 as the encoder
        resnet = models.resnet34(pretrained=True)

        self.encoder_layers = list(resnet.children())[:8]  
        self.encoder1 = nn.Sequential(*self.encoder_layers[:3])  
        self.encoder2 = nn.Sequential(*self.encoder_layers[3:5]) 
        self.encoder3 = nn.Sequential(*self.encoder_layers[5:6]) 
        self.encoder4 = nn.Sequential(*self.encoder_layers[6:7])
        self.encoder5 = nn.Sequential(*self.encoder_layers[7:8]) 
        
        # Decoder blocks with upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 16x16x256
        self.decoder1 = self.decoder_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32x32x128
        self.decoder2 = self.decoder_block(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 64x64x64
        self.decoder3 = self.decoder_block(128, 64)
        
        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 128x128x64
        self.decoder4 = self.decoder_block(128, 64)
        
        self.upconv5 = nn.ConvTranspose2d(64, n_classes, kernel_size=2, stride=2)  # 256x256xout_channels
        self.sigmoid = nn.Sigmoid()
        
    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 128x128x64
        e2 = self.encoder2(e1)  # 64x64x64
        e3 = self.encoder3(e2)  # 32x32x128
        e4 = self.encoder4(e3)  # 16x16x256
        e5 = self.encoder5(e4)  # 8x8x512
        
        # Decoder
        d1 = self.upconv1(e5)  # 16x16x256
        d1 = torch.cat([d1, e4], dim=1)  # Skip connection
        d1 = self.decoder1(d1)
        
        d2 = self.upconv2(d1)  # 32x32x128
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.decoder2(d2)
        
        d3 = self.upconv3(d2)  # 64x64x64
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d4 = self.upconv4(d3)  # 128x128x64
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.decoder4(d4)
        
        d5 = self.upconv5(d4)  # 256x256xout_channels
        
        out = self.sigmoid(d5)
        return out


class UNet50(nn.Module):
    def __init__(self, n_classes=1, dropout = 0.2, pretrained=True):
        super(UNet50, self).__init__()
        
        # Load a pre-trained ResNet-50 model as the encoder
        resnet = models.resnet50(pretrained=pretrained)
        
        # Encoder layers: we take the outputs at different levels
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # Output after 1st block
        self.encoder2 = resnet.layer2  # Output after 2nd block
        self.encoder3 = resnet.layer3  # Output after 3rd block
        self.encoder4 = resnet.layer4  # Output after 4th block

        # Decoder layers
        self.decoder4 = self.conv_block(2048, 1024)  # After last ResNet block
        self.decoder3 = self.conv_block(1024 + 1024, 512)
        self.decoder2 = self.conv_block(512 + 512, 256)
        self.decoder1 = self.conv_block(256 + 256, 64)
        self.decoder0 = self.conv_block(64 + 64, 64)

        # Final convolution to produce the segmentation map
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Decoder with skip connections
        dec4 = F.interpolate(self.decoder4(enc4), scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = F.interpolate(self.decoder3(torch.cat([dec4, enc3], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = F.interpolate(self.decoder2(torch.cat([dec3, enc2], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = F.interpolate(self.decoder1(torch.cat([dec2, enc1], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        dec0 = F.interpolate(self.decoder0(torch.cat([dec1, enc0], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        return self.final_conv(dec0)