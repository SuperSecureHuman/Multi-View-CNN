import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            split (string): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []  # This will contain paths to all views of all samples
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name, self.split)
            sample_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            
            # Extract unique indices from the file names
            sample_indices = list(set([f.split('.obj.shaded_')[0].split('_')[-1] for f in sample_files]))
            
            for index in sample_indices:
                # Use string formatting to create the correct file name
                views = [os.path.join(class_dir, f"{class_name}_{index}.obj.shaded_v{str(view_num).zfill(3)}.png") 
                         for view_num in range(1, len(sample_files) // len(sample_indices) + 1)]
                self.samples.append((views, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        views, label = self.samples[idx]
        images = [Image.open(view_path).convert('RGB') for view_path in views]
        
        if self.transform:
            images = [self.transform(image) for image in images]
        
        # Stack images to create a tensor of shape [V, C, H, W]
        images = torch.stack(images, dim=0)
        return images, label
