import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
import cv2
import numpy as np
preprocess_images = transforms.Compose([
transforms.Resize((256,256)),
transforms.Lambda(lambda img: transforms.functional.equalize(img) ),
transforms.Lambda(lambda img: transforms.functional.autocontrast(img) ),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_labels = transforms.Compose([
transforms.Resize((256,256)),
transforms.Lambda(lambda img: transforms.functional.equalize(img) ),
transforms.Lambda(lambda img: transforms.functional.autocontrast(img) ),
transforms.ToTensor(),
])

folders = ['train', 'val', 'test']


class LeafDataset(Dataset):
    """
    Custom Dataset

        """
    def __init__(self, image_dir, sub=''):
        """
        Arguments:
            image dir: string
                indicates where is the dataset
            sub: string
                subdirectory name at default there is no subdirectory
        """
        
        self.sub=sub
        self.image_dir = os.path.join(image_dir, sub)
        self.images = []
        self.labels = []

        for img_path in glob.glob(self.image_dir + '/*'):
            if 'mask' in img_path:
                self.labels.append(img_path)

            else:
                self.images.append(img_path)
        self.images.sort()
        self.labels.sort()
        print(len(self.images))
        print(len(self.labels))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image = Image.open(img_path)
        label = Image.open(label_path).convert('L')

        image_tensor = preprocess_images(image)
        label_tensor = preprocess_labels(label)
        #mapping tensor between 0 and 1 
        label_tensor = label_tensor / np.amax(label_tensor.numpy())
        return image_tensor, label_tensor
    

