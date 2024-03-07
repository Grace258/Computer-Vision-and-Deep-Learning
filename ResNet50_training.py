import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
import os
import torch.nn.functional as F
from torchvision import ops
from ResNet50 import CustomResNet50, lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create an instance of the custom model
model = CustomResNet50()
model = model.to(device)

# Define loss function, optimizer, and scheduler
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def Load_data_withRE():                     #Random Erasing part
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),  
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),                
        transforms.RandomErasing(),         #check HERE!!!
    ])  

    train_data = datasets.ImageFolder("C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/training_dataset", transform=transform)
    val_data = datasets.ImageFolder("C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/validation_dataset", transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16, shuffle=True, num_workers = 4)  
    

    return train_loader, test_loader

def Load_data_withoutRE():
    transform = transforms.Compose([ 
        transforms.CenterCrop(224),       
        transforms.Resize(224), 
        transforms.ToTensor(), 
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),        
              
    ])

    train_data = datasets.ImageFolder("C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/training_dataset", transform=transform)
    val_data = datasets.ImageFolder("C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/validation_dataset", transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16, shuffle=True, num_workers = 4)

    return train_loader, test_loader

def binary_cross_entropy(target_pred, target):
    cross_entropy = F.binary_cross_entropy(target_pred, target)
    return cross_entropy

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

def train_and_get_accuracy(train_loader, test_loader, mode, num_epochs=3):    
    print(f'Training device: {device}')     
    best_accuracy = 0.0  # Initialize with a low value   
    
    for epoch in range(num_epochs):
        model.train()               

        for batch_idx, (images, labels) in enumerate (train_loader):            
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(0).float()            
            optimizer.zero_grad()            
            logits = model(images)
            outputs = torch.sigmoid(logits)
            
            loss = binary_cross_entropy(outputs, labels.unsqueeze(1))            
            loss.backward() 
            optimizer.step()             

            if not batch_idx % 300:            
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Train Loss: {loss:.4f}') 
        
        model.eval()               
        
        with torch.no_grad():
            train_accuracy = compute_accuracy(model, train_loader, device)
            val_accuracy = compute_accuracy(model, test_loader, device)

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} Train Acc.: {train_accuracy: .3f}%'
              f' | Validations Acc.: {val_accuracy:.3f}%')           
                        

        # Check if the current accuracy is the best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_dir = "C:/Users/yingc/Desktop/cvdl_hw2"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if mode == 0:
                torch.save(model, os.path.join(save_dir, "best_RE_model.pth"))
            else:
                torch.save(model, os.path.join(save_dir, "best_withoutRE_model.pth"))
    print('Training complete.')   

    return best_accuracy

def generate_plot(accuracy_withRE, accuracy_withoutRE):    
    classes = ['Without Random Erasing', 'With Random Erasing']
    
    bar = plt.bar(classes, [accuracy_withRE, accuracy_withoutRE])
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy Comparison")

    for item in bar:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2.,
            height*1.,
            '%d' % int(height),
            ha = "center",
            va = "bottom"
        )
    
    plt.savefig('C:/Users/yingc/Desktop/cvdl_hw2/accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    train_loader_withRE, test_loader_withRE = Load_data_withRE()
    train_loader_withoutRE, test_loader_withoutRE = Load_data_withoutRE()

    accuracies_withRE = train_and_get_accuracy(train_loader_withRE, test_loader_withRE, mode = 0).cpu().numpy()
    accuracies_withoutRE = train_and_get_accuracy(train_loader_withoutRE, test_loader_withoutRE, mode = 1).cpu().numpy()

    generate_plot(accuracies_withRE, accuracies_withoutRE)