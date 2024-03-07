import time
import os

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import ops
from torchvision import models
from torchsummary import summary

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/training_dataset"
val_dir = "C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/validation_dataset"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 244)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16, shuffle=True)
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.squeeze(0).float()
        targets = targets.to(device)
        logits = model(features)

        probas = torch.sigmoid(logits)
        predicted_labels = (probas > torch.tensor([0.5]).to(device)).float()*1
        predicted_labels = predicted_labels.transpose(0, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return (correct_pred.float()/num_examples * 100)

from torchvision.models import resnet50, ResNet50_Weights

for features, targets in train_loader:
    break

model = resnet50(weights=ResNet50_Weights.DEFAULT)

for params in model.parameters():
    params.requires_grad_ = False

nr_filters = model.fc.in_features
model.fc = nn.Linear(nr_filters, 1)

NUM_EPOCHS = 1
optimizer = torch.optim.Adam(model.parameters())

start_time = time.time()
train_acc_1st, valid_acc_FL = [], []

def sigmoid_focal_loss(target_pred, target):
    focal_loss = ops.sigmoid_focal_loss(torch.sigmoid(target_pred), target, 
                                    alpha=0.4, gamma=1.0, reduction='mean')
    return focal_loss

model_fl = model.to(DEVICE)    

for epoch in range(NUM_EPOCHS):
    model_fl.train()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.squeeze(0).float()
        targets = targets.to(DEVICE)

        logits = model_fl(features)
        probas = torch.sigmoid(logits)

        logits = logits.squeeze(1).float()
        
        loss = sigmoid_focal_loss(logits, targets)
        loss.backward()

        optimizer.step()  
        optimizer.zero_grad()
        
        if not batch_idx % 300:
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} |'
                  f'Batch: {batch_idx:03d}/{len(train_loader):03d} |'
                  f'Cost: {loss:.4f}')
    
    model_fl.eval()
    
    with torch.no_grad():
        train_acc = compute_accuracy(model_fl, train_loader, device=DEVICE)
        valid_acc = compute_accuracy(model_fl, val_loader, device=DEVICE)
        train_acc_1st.append(float(train_acc))
        valid_acc_FL.append(float(valid_acc))

        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc: .3f}%'
              f' | Validations Acc.: {valid_acc:.3f}%')

    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')
path = 'C:/Users/yingc/Desktop/cvdl_hw2/model_fl_record.pt'
torch.save(model_fl, path)