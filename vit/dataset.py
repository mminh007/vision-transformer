import os 
import pandas as pd
from torch.utils.data import Dataset
import torch
import cv2

class ImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None, train=True):
        self.img_labels = pd.read.csv(annotation_file)
        self.type = "/train" if train else "/valid"
        self.img_dir = img_dir + self.type
        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)

        label = self.img_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    