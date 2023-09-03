# Multi View Convolutional Neural Networks for 3D Shape Recognition

Possibly the simplest implementation of MVCNN in PyTorch.

The backbone pretrained network is ResNet18 from TIMM. It can be replaced with anything else. You need to change the following in `mvcnn_resnet18.py`:

```python
class MVCNN(nn.Module):
    def __init__(self, num_classes):
        super(MVCNN, self).__init__()
        
        ##### CHANGE THIS #####
        self.resnet = timm.create_model('resnet18', pretrained=True)
        self.resnet.global_pool = nn.Identity()  # Remove global pooling
        self.resnet.fc = nn.Identity()           # Remove fully connected layer

        # Ensure ResNet parameters are not trainable
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #### CHANGE THE PARAMS OF THE CONV LAYER ####
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #### CHANGE THE PARAMS OF THE FC LAYER ####
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
```

## Data Loading

Make sure the data is in the format:

```
Batch x Views x Channels x Height x Width
```

For example, if you have 12 views of a 3-channel image of size 224x224, the shape of the input should be:

```
(1, 12, 3, 224, 224)
```

Checkout `dataset_modelnet40.py` for an example of how to load the data.

## Training

Training loop defined in `train.py`. You can change the optimizer, learning rate scheduler, loss function, etc. in this file.

