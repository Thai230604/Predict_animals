from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose, RandomAffine, ColorJitter, Normalize, InterpolationMode
import os
from PIL import Image

class Animal(Dataset):
    def __init__(self, root, train=True):
        mode = 'train' if train else 'test' 
        self.root = os.path.join(root, mode)

        self.image_paths = []  
        self.labels = []  

        for i, path in enumerate(os.listdir(self.root)):
            for image_path in os.listdir(os.path.join(self.root, path)):
                self.image_paths.append(os.path.join(self.root, path, image_path))
                self.labels.append(i)

        self.train = train  

        self.train_transforms = Compose([
            Resize((224, 224)),
            RandomAffine(
                degrees=(-15, 15),  
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,  
                fill=[0.485 * 255, 0.456 * 255, 0.406 * 255] 
            ),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)  

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])  
        image = image.convert('RGB')
        if self.train:  
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)
        return image, self.labels[idx]  