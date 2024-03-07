import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VGG19 import VGG19BN  # Assuming you have a VGG19 architecture with batch normalization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def load_mnist():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3801,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_loader, test_loader

def to_percentage(y, _):
    return f'{y*100:.0f}%'

def train_vgg19_bn_mnist(train_loader, test_loader, epochs=30, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VGG19BN(num_classes=10).to(device)  # Assuming 10 classes for MNIST
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    # Lists to store training and validation metrics
    train_loss_list, train_accuracy_list = [], []
    val_loss_list, val_accuracy_list = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            logits, probas = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Train Loss: {total_loss / (batch_idx + 1):.4f}, Train Accuracy: {correct / total:.4f}')

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)

        model.eval()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            loss = criterion(logits, target)

            _, predicted_test = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted_test == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Test Batch {batch_idx}/{len(test_loader)}, '
                      f'Test Accuracy: {correct / total:.4f}')

        val_loss = loss.item()
        val_accuracy = correct / total
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        # Save the model if it has the highest accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model, 'C:/Users/yingc/Desktop/cvdl_hw2/best_model_mnist.pth')

    print('Training complete.')

    # Save the training and validation metrics plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label='train_acc')
    plt.plot(val_accuracy_list, label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))

    plt.savefig('C:/Users/yingc/Desktop/cvdl_hw2/training_plot_mnist.png')
    plt.show()

if __name__ == "__main__":
    train_loader, test_loader = load_mnist()
    train_vgg19_bn_mnist(train_loader, test_loader)
