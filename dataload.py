## DataLoader
import glob
import os

import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
from PIL import Image


class PolypLoader(Dataset):
    """
    Creates a Class to Load the Polyp Dataset for use with Pytorch Dataloader
    
    
    """
    
    def __init__(self, path ,transform = None):
        """
        """
        self.path = path
        self.train = os.path.join(path, 'train')
        self.label = os.path.join(path, 'label')
        
        assert (len(os.listdir(self.train)) == len(os.listdir(self.label))) ## assert same length train and test labels
        
        ## Get FileNames
        file_names = os.listdir(self.train)
        self.file_names = [os.path.splitext(x)[0] for x in file_names]
        
        ## Store Full File Paths for Image
        train_str = '%s.tif'
        label_str = '%s.tif'
        self.train_files = [os.path.join(self.train, train_str%name) for name in self.file_names]
        self.label_files = [os.path.join(self.label, label_str%name) for name in self.file_names]
        
        ## Transforms
        self.num = len(self.file_names)
        
        ## Preloading Images
        if transform is not None:
            self.transform = transforms.Compose(transform)
        else:
            self.transform = None
               
        
        
    def __getitem__(self, index):
        """
        Returns 
        """
        #assert(index < self.num)
        #assert(index >= 0)
        
        ## Loading Images with cv2
        target = cv2.cvtColor(cv2.imread(self.train_files[index]), cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_files[index])

        ## Convert Images to PIL Format
        target = Image.fromarray(target)
        label = Image.fromarray(label[:,:,0])
        #label = label[:,:,0] ## Note Label is 3 channel tiff with all channels same
        
        
        if self.transform is not None:
            target = self.transform(target)
            label = self.transform(label)
            
        return target, label
        
    def __len__(self):
        """
        Returns Length of Dataset
        """
        return self.num
