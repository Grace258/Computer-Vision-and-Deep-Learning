import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all layers except the final classification layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the final classification layer
        num_ftr = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftr, 1),  
            
        )

    def forward(self, x):
        return self.model(x)


