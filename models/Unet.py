## Define Unet Model

## Import Statements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#from torchsummary import summary


## Helper Functions and Classes
def Conv_BNN(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
  return[
         nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, 
                   padding = padding, stride = stride),
         nn.BatchNorm2d(out_channels),
         nn.ReLU(inplace=True)]

class Interpolate(nn.Module):
  """
  Defines Interpolate function (weird inputs)
  """
  def __init__(self,scale_factor =2, mode = 'bilinear', align_corners = False):
    super(Interpolate, self).__init__()
    self.mode = mode
    self.align_corners = align_corners
    self.scale_factor = scale_factor
    
  
  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale_factor, mode = 'bilinear',
                         align_corners = self.align_corners)
    return x

class Decode_BNN(nn.Module):
  def __init__(self,in_channels, out_channels, kernel_size=1, padding=0, stride=1):
    super(Decode_BNN, self).__init__()
    self.interp = Interpolate()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                     padding = padding, stride = stride)
    self.relu = nn.ReLU(inplace = True)

  def forward(self, x):
    x = self.interp(x)
    x = self.conv(x)
    x = self.relu(x)
    return x
  

## Unet Main Class

class Unet(nn.Module):
  """
  Unet from scratch.
  """
  
  def __init__(self, inshape):
    super(Unet, self).__init__()
    num_classes = 1
    self.in_channels, self.height, self.width = inshape
    
    ## Define Pooling Layer
    self.pool = nn.MaxPool2d(2,2)
    
    ## Define Convolution Layers
    self.conv1 = Conv_BNN(self.in_channels, 64)
    self.conv2 = Conv_BNN(64,64)
    
    self.conv3 = Conv_BNN(64,128)
    self.conv4 = Conv_BNN(128,128)
   
    self.conv5 = Conv_BNN(128, 256)
    self.conv6 = Conv_BNN(256,256)
    
    self.conv7 = Conv_BNN(256,512)
    self.conv8 = Conv_BNN(512,512)
    
    self.conv9 = Conv_BNN(512, 1024)
    self.conv10 = Conv_BNN(1024,1024)
    
    ## Define Down Convolution Layers
    self.down1 = nn.Sequential(*self.conv1, *self.conv2)
    self.down2 = nn.Sequential(*self.conv3, *self.conv4)
    self.down3 = nn.Sequential(*self.conv5, *self.conv6)
    self.down4 = nn.Sequential(*self.conv7, *self.conv8)
    self.down5 = nn.Sequential(*self.conv9, *self.conv10)
    
    ## 2x2 Up Conv
    self.upconv1 = Decode_BNN(1024,512)
    self.upconv2 = Decode_BNN(512,256)
    self.upconv3 = Decode_BNN(256,128)
    self.upconv4 = Decode_BNN(128,64)
    
    
    ## Up Convolution Layer
    self.conv11 =  Conv_BNN(1024, 512)
    self.conv12 = Conv_BNN(512,512)
    
    self.conv13 = Conv_BNN(512, 256)
    self.conv14 = Conv_BNN(256,256)
    
    self.conv15 = Conv_BNN(256,128)
    self.conv16 = Conv_BNN(128, 128)
    
    self.conv17 = Conv_BNN(128, 64)
    self.conv18 = Conv_BNN(64,64)
    

    
    
    ## Define UpSampling Layers
    self.up1 = nn.Sequential(*self.conv11, *self.conv12)
    self.up2 = nn.Sequential(*self.conv13, *self.conv14)
    self.up3 = nn.Sequential(*self.conv15, *self.conv16)
    self.up4 = nn.Sequential(*self.conv17, *self.conv18)
    
    ## Define Classifier
    self.classify = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0 )
    
    return
  
    
  def forward(self, x):
    
    ## Down Convlolution Blocks
    layer1 = self.down1(x)
    pool1 = self.pool(layer1)
    
    layer2 = self.down2(pool1)
    pool2 = self.pool(layer2)
    
    layer3 = self.down3(pool2)
    pool3 = self.pool(layer3)
    
    layer4 = self.down4(pool3)
    pool4 = self.pool(layer4)
    
    layer5 = self.down5(pool4)
    
    ## Up Sampling Blocks
    up1 = self.upconv1(layer5)
    cat1 = torch.cat([up1, layer4],1)
    out = self.up1(cat1)
    
    up2 = self.upconv2(out)
    cat2 = torch.cat([up2, layer3],1)
    out = self.up2(cat2)
    
    up3 = self.upconv3(out)
    cat3 = torch.cat([up3, layer2],1)
    out = self.up3(cat3)
    
    up4 = self.upconv4(out)
    cat4 = torch.cat([up4, layer1],1)
    up4_out = self.up4(cat4)
    
    
    ## 
    class_out = self.classify(up4_out)
    out = F.sigmoid(class_out)
    
    return out
