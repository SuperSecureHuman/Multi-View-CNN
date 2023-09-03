import timm
import torch.nn as nn

class MVCNN(nn.Module):
    def __init__(self, num_classes):
        super(MVCNN, self).__init__()
        
        # Load pre-trained ResNet model from timm and remove its classification head
        self.resnet = timm.create_model('resnet18', pretrained=True)
        self.resnet.global_pool = nn.Identity()  # Remove global pooling
        self.resnet.fc = nn.Identity()           # Remove fully connected layer

        # Ensure ResNet parameters are not trainable
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # CNN after pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, views):
        # views should be of shape [B, V, C, H, W]
        # where B is batch size, V is number of views, C is channels, H is height, and W is width
        
        B, V, C, H, W = views.shape
        views = views.view(B*V, C, H, W)  # Reshape for processing

        # Get features for each view
        features = self.resnet(views)

        # Reshape features to [B, V, C', H', W']
        features = features.view(B, V, features.shape[1], features.shape[2], features.shape[3])
        
        # Pool across views
        pooled_features, _ = features.max(dim=1)  # This is the view pooling

        x = self.cnn(pooled_features)
        x = self.classifier(x)
        
        return x
