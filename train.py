import torch
import torch.nn as nn

from dataset_modelnet40 import MultiViewDataset
from mvcnn_resnet18 import MVCNN
from utils import train_one_epoch, evaluate

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = MultiViewDataset(root_dir='/home/venom/ssd/3d/modelnet40_images_new_12x', split='train', transform=transform)
test_dataset = MultiViewDataset(root_dir='/home/venom/ssd/3d/modelnet40_images_new_12x', split='test', transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


# Initialize model, criterion, and optimizer
model = MVCNN(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with evaluation
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    test_accuracy = evaluate(model, test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


torch.save(model.state_dict(), 'mvcnn_resnet18.pth')