# %%
import torch.nn as nn
import torch

# %%
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm.notebook import tqdm


# %% [markdown]
# ### Dataset Preparation (MNIST Handwritten)

# %%
# Define transformations for MNIST Handwritten
transform_mnist = transforms.Compose([
    transforms.RandomRotation(degrees=10), # random rotation for data augmentation
    transforms.Pad(2), # LeNet expects 32x32 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST Handwritten training and test datasets
trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

# Create data loaders
batch_size = 512
trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=batch_size, shuffle=True)
testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=batch_size, shuffle=False)

print(f"MNIST Handwritten training samples: {len(trainset_mnist)}")
print(f"MNIST Handwritten test samples: {len(testset_mnist)}")

# %% [markdown]
# ### Install MedMNIST

# %%
!pip install medmnist


# %% [markdown]
# ### Dataset Preparation (PneumoniaMNIST)

# %%
import medmnist
from medmnist import INFO, Evaluator

# Define transformations for PneumoniaMNIST
# Note: MedMNIST datasets are typically 28x28, so resize to 32x32 for LeNet
transform_pneumonia = transforms.Compose([
    transforms.ColorJitter(contrast=0.5),
    transforms.Pad(2), # LeNet expects 32x32 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]) # Common normalization for image data
])

# Load PneumoniaMNIST training and test datasets
data_flag = 'pneumoniamnist'
info = INFO[data_flag]

# MedMNIST datasets are typically 1 channel (grayscale) and have num_classes for output_size
# We also need to reshape the labels from (N, 1) to (N,) for CrossEntropyLoss

trainset_pneumonia = medmnist.PneumoniaMNIST(split='train', transform=transform_pneumonia, download=True)
trainset_pneumonia.labels = trainset_pneumonia.labels.squeeze()
testset_pneumonia = medmnist.PneumoniaMNIST(split='test', transform=transform_pneumonia, download=True)
testset_pneumonia.labels = testset_pneumonia.labels.squeeze()

# Create data loaders
trainloader_pneumonia = torch.utils.data.DataLoader(trainset_pneumonia, batch_size=batch_size, shuffle=True)
testloader_pneumonia = torch.utils.data.DataLoader(testset_pneumonia, batch_size=batch_size, shuffle=False)

print(f"PneumoniaMNIST training samples: {len(trainset_pneumonia)}")
print(f"PneumoniaMNIST test samples: {len(testset_pneumonia)}")

# %% [markdown]
# ### Dataset Preparation (MNIST Fashion)

# %%
# Define transformations for MNIST Fashion (same as Handwritten for input size consistency)
transform_fashion = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.Pad(2), # LeNet expects 32x32 input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Common normalization for Fashion MNIST
])

# Load MNIST Fashion training and test datasets
trainset_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fashion)
testset_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fashion)

# Create data loaders
trainloader_fashion = torch.utils.data.DataLoader(trainset_fashion, batch_size=batch_size, shuffle=True)
testloader_fashion = torch.utils.data.DataLoader(testset_fashion, batch_size=batch_size, shuffle=False)

print(f"MNIST Fashion training samples: {len(trainset_fashion)}")
print(f"MNIST Fashion test samples: {len(testset_fashion)}")

# %% [markdown]
# ### Training and Evaluation Functions

# %%
def train_model(model, trainloader, criterion, optimizer, num_epochs=10, device='cpu', scheduler=None):
    model.train()
    model.to(device)
    history = {'loss': [], 'acc': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, LR: {current_lr}')

        if scheduler is not None:
            scheduler.step()

    return history

# %%
def evaluate_model(model, testloader, device='cpu', print_metrics=True):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in tqdm(testloader, desc="[Evaluating]", leave=False):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    precision = precision_score(all_labels, all_predicted, average='weighted')

    if print_metrics:
        print(f'Accuracy on the test set: {accuracy:.2f}%')
        print(f'F1-Score (weighted): {f1:.2f}')
        print(f'Recall (weighted): {recall:.2f}')
        print(f'Precision (weighted): {precision:.2f}')

    return accuracy, f1, recall, precision

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def plot_history(history, title="Training History"):
    """Vẽ biểu đồ Loss và Accuracy từ lịch sử huấn luyện"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history['loss'], color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(history['acc'], color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(title)
    fig.tight_layout()
    plt.show()

def plot_confusion_matrix(model, testloader, device, classes, title="Confusion Matrix"):
    """Dự đoán và vẽ ma trận nhầm lẫn"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# %%
# changing the lenet with newer code style -> increase nums of params but it may be better, and have no different about speed

class Lenet(nn.Module):
  def __init__(self, input_size, output_size): # the input size should be 32 with padding
    super(Lenet, self).__init__()

    self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
    self.bn1 = nn.BatchNorm2d(6) # BN cho 6 kênh của C1
    self.s2 = nn.MaxPool2d(2, 2)
    self.c3 = nn.Conv2d(6, 16, kernel_size=5,stride=1)
    self.bn3 = nn.BatchNorm2d(16) # BN cho 16 kênh sau khi cat ở C3
    self.s4 = nn.MaxPool2d(2, 2)
    self.c5 = nn.Conv2d(16, 120, stride=1, kernel_size=5)
    self.bn5 = nn.BatchNorm2d(120)
    self.fc6 = nn.Linear(120, 84)
    self.bn6 = nn.BatchNorm1d(84)

    self.output = nn.Linear(84, output_size)

  def forward(self, x):

    # Inserts a dimension with a size of one (a singleton dimension) into the tensor's shape.
    if x.dim() == 3:
      x = x.unsqueeze(0)

    x = self.bn1(self.c1(x))
    x = torch.relu(self.s2(x))
    x = self.bn3(self.c3(x))
    x = torch.relu(self.s4(x))
    x = self.bn5(self.c5(x))

    # x is now of shape (N, 120, 1, 1) and need to be at shape (N, 120) to proceed so we use x.view

    x = x.view(x.shape[0], -1)

    x = torch.relu(self.bn6(self.fc6(x)))

    return self.output(x)


# %%
# changing the lenet with newer code style -> increase nums of params but it may be better, and have no different about speed

class LenetDropout(nn.Module):
  def __init__(self, input_size, output_size): # the input size should be 32 with padding
    super(LenetDropout, self).__init__()

    self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
    self.bn1 = nn.BatchNorm2d(6) # BN cho 6 kênh của C1
    self.s2 = nn.MaxPool2d(2, 2)
    self.c3 = nn.Conv2d(6, 16, kernel_size=5,stride=1)
    self.bn3 = nn.BatchNorm2d(16) # BN cho 16 kênh sau khi cat ở C3
    self.s4 = nn.MaxPool2d(2, 2)
    self.c5 = nn.Conv2d(16, 120, stride=1, kernel_size=5)
    self.bn5 = nn.BatchNorm2d(120)
    self.fc6 = nn.Linear(120, 84)
    self.bn6 = nn.BatchNorm1d(84)

    self.output = nn.Linear(84, output_size)

    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    if x.dim() == 3:
        x = x.unsqueeze(0)

    # --- Khối 1 ---
    x = self.c1(x)
    x = self.bn1(x)
    x = torch.relu(x)      # Kích hoạt (lọc nhiễu âm) TRƯỚC
    x = self.s2(x)         # Pooling (nén đặc trưng) SAU

    # --- Khối 2 ---
    x = self.c3(x)
    x = self.bn3(x)
    x = torch.relu(x)      # Kích hoạt TRƯỚC
    x = self.s4(x)         # Pooling SAU

    # --- Khối 3 ---
    x = self.c5(x)
    x = self.bn5(x)
    x = torch.relu(x)

    # --- Flatten ---
    x = x.view(x.shape[0], -1)

    # --- Khối FC ---
    x = self.fc6(x)
    x = self.bn6(x)
    x = torch.relu(x)
    x = self.dropout(x)    # Dropout 0.5 hoạt động cực tốt ở đây

    return self.output(x)

# %%
from sklearn.metrics import f1_score, recall_score, precision_score

# %% [markdown]
# ### LeNet on MNIST Handwritten

# %%
print("\n--- Training LeNet on MNIST Handwritten ---")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lenet_mnist = Lenet(input_size=32, output_size=10)
criterion_mnist = nn.CrossEntropyLoss()
optimizer_mnist = optim.Adam(lenet_mnist.parameters(), lr=0.01)
# Khởi tạo scheduler
scheduler_mnist = optim.lr_scheduler.StepLR(optimizer_mnist, step_size=10, gamma=0.1)

# Huấn luyện với scheduler
history_mnist = train_model(lenet_mnist, trainloader_mnist, criterion_mnist, optimizer_mnist, num_epochs=30, device=device, scheduler=scheduler_mnist)
plot_history(history_mnist, title="MNIST Handwritten Training History")

classes_mnist = [str(i) for i in range(10)]
plot_confusion_matrix(lenet_mnist, testloader_mnist, device, classes_mnist, title="MNIST Handwritten Confusion Matrix")
accuracy_mnist, f1_mnist, recall_mnist, precision_mnist = evaluate_model(lenet_mnist, testloader_mnist, device=device)

# %%
accuracy_mnist, f1_mnist, recall_mnist, precision_mnist = evaluate_model(lenet_mnist, testloader_mnist, device=device)

# %% [markdown]
# ### LeNet on MNIST Fashion

# %%
import torch
import torch.nn as nn
import torch.optim as optim
# LeNetWide class definition with fix
class LeNetWide(nn.Module):
    def __init__(self, output_size=10):
        super(LeNetWide, self).__init__()

        # --- Khối 1: Tăng từ 6 lên 32 filters ---
        # Input: 28x28. Thêm padding=2 để ảnh lên 32x32 -> Output: 32 channels, 28x28
        self.c1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.s2 = nn.MaxPool2d(2, 2) # Output: 32 channels, 14x14

        # --- Khối 2: Tăng từ 16 lên 64 filters ---
        # Dùng Conv2d tiêu chuẩn thay vì c3_indices phức tạp
        self.c3 = nn.Conv2d(32, 64, kernel_size=5) # Output: 64 channels, 10x10
        self.bn3 = nn.BatchNorm2d(64)
        self.s4 = nn.MaxPool2d(2, 2) # Output: 64 channels, 5x5

        # --- Khối 3: Tăng từ 120 lên 256 filters ---
        self.c5 = nn.Conv2d(64, 256, kernel_size=5) # Output: 256 channels, 1x1
        self.bn5 = nn.BatchNorm2d(256)

        # --- Khối Fully Connected: Tăng từ 84 lên 128 ---
        # FIX: Changed input features from 256 to 1024
        self.fc6 = nn.Linear(1024, 128)
        self.bn6 = nn.BatchNorm1d(128)

        self.output = nn.Linear(128, output_size)

        # Dropout 0.5 để chống overfit do lượng tham số đã tăng lên nhiều
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Khối 1
        x = self.c1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.s2(x)

        # Khối 2
        x = self.c3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.s4(x)

        # Khối 3
        x = self.c5(x)
        x = self.bn5(x)
        x = torch.relu(x)

        # Flatten [Batch, 256, 1, 1] -> [Batch, 256]
        x = x.view(x.size(0), -1)

        # Khối FC
        x = self.fc6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.output(x)

# %%


print("\n--- Training LeNet on MNIST Fashion ---")
lenet_fashion = LeNetWide(output_size=10)
criterion_fashion = nn.CrossEntropyLoss()
optimizer_fashion = optim.Adam(lenet_fashion.parameters(), lr=0.001)
scheduler_fashion = optim.lr_scheduler.StepLR(optimizer_fashion, step_size=10, gamma=0.1)

history_fashion = train_model(lenet_fashion, trainloader_fashion, criterion_fashion, optimizer_fashion, num_epochs=30, device=device, scheduler=scheduler_fashion)
plot_history(history_fashion, title="MNIST Fashion Training History")

classes_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot_confusion_matrix(lenet_fashion, testloader_fashion, device, classes_fashion, title="MNIST Fashion Confusion Matrix")
accuracy_fashion, f1_fashion, recall_fashion, precision_fashion = evaluate_model(lenet_fashion, testloader_fashion, device=device)

# %%
import torch
import torch.nn as nn

class LeNetWide_3x3(nn.Module):
    def __init__(self, output_size=10):
        super(LeNetWide_3x3, self).__init__()

        # Khối 1: Dùng kernel 3x3, padding 1
        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.s2 = nn.MaxPool2d(2, 2)

        # Khối 2: Tăng filter lên 64, kernel 3x3
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.s4 = nn.MaxPool2d(2, 2)

        # Khối 3: Tăng filter lên 128, kernel 3x3
        self.c5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        # ĐIỂM SÁNG: Ép đầu ra của C5 luôn về kích thước 2x2 bất kể ảnh gốc như thế nào
        self.pool_final = nn.AdaptiveAvgPool2d((2, 2))

        # FC Layer: 128 kênh x kích thước 2x2 = 512
        self.fc6 = nn.Linear(128 * 2 * 2, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

        self.output = nn.Linear(128, output_size)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)

        # Block 1
        x = self.s2(torch.relu(self.bn1(self.c1(x))))
        # Block 2
        x = self.s4(torch.relu(self.bn3(self.c3(x))))
        # Block 3
        x = torch.relu(self.bn5(self.c5(x)))

        # Ép về 2x2
        x = self.pool_final(x)

        # Flatten (Batch_size, 128 * 2 * 2)
        x = x.view(x.size(0), -1)

        # FC Block
        x = self.dropout(torch.relu(self.bn6(self.fc6(x))))

        return self.output(x)

# %%


print("\n--- Training LeNet on MNIST Fashion ---")
lenet_fashion = LeNetWide_3x3(output_size=10)
criterion_fashion = nn.CrossEntropyLoss()
optimizer_fashion = optim.Adam(lenet_fashion.parameters(), lr=0.001)
scheduler_fashion = optim.lr_scheduler.StepLR(optimizer_fashion, step_size=10, gamma=0.1)

history_fashion = train_model(lenet_fashion, trainloader_fashion, criterion_fashion, optimizer_fashion, num_epochs=30, device=device, scheduler=scheduler_fashion)
plot_history(history_fashion, title="MNIST Fashion Training History")

classes_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot_confusion_matrix(lenet_fashion, testloader_fashion, device, classes_fashion, title="MNIST Fashion Confusion Matrix")
accuracy_fashion, f1_fashion, recall_fashion, precision_fashion = evaluate_model(lenet_fashion, testloader_fashion, device=device)

# %% [markdown]
# ### LeNet on PneumoniaMNIST

# %%
print("\n--- Training LeNet on PneumoniaMNIST ---")
# Calculate class weights for PneumoniaMNIST
class_counts = torch.bincount(torch.tensor(trainset_pneumonia.labels))
total_samples = class_counts.sum().item()
num_classes = len(class_counts)
class_weights = total_samples / (num_classes * class_counts.float())
class_weights = class_weights.to(device)

lenet_pneumonia = LeNetWide_3x3(input_size=32, output_size=2)
criterion_pneumonia = nn.CrossEntropyLoss(weight=class_weights) # Apply weighted loss
optimizer_pneumonia = optim.Adam(lenet_pneumonia.parameters(), lr=0.01)
scheduler_pneumonia = optim.lr_scheduler.StepLR(optimizer_pneumonia, step_size=10, gamma=0.1)

history_pneumonia = train_model(lenet_pneumonia, trainloader_pneumonia, criterion_pneumonia, optimizer_pneumonia, num_epochs=30, device=device, scheduler=scheduler_pneumonia)
plot_history(history_pneumonia, title="PneumoniaMNIST Training History")

classes_pneumonia = ['Normal', 'Pneumonia']
plot_confusion_matrix(lenet_pneumonia, testloader_pneumonia, device, classes_pneumonia, title="PneumoniaMNIST Confusion Matrix")
accuracy_pneumonia, f1_pneumonia, recall_pneumonia, precision_pneumonia = evaluate_model(lenet_pneumonia, testloader_pneumonia, device=device)


