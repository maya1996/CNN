
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_handling():

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = CIFAR10(root='./data',train=True,download=True,transform=transform_train)
    test_data = CIFAR10(root='./data',train=False,download=True,transform=transform_val)

    training_size = len(training_data) - 10000
    validation_size = len(training_data) - training_size
    train_batch, validation_batch = random_split(training_data, [training_size, validation_size])

    train = DataLoader(train_batch, batch_size=32, shuffle=True, num_workers=2)
    validation = DataLoader(validation_batch, batch_size=32, shuffle=True, num_workers=2)
    test = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

    return train, validation, test

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.4)


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(f.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(f.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(f.relu(x))
        x = self.bn4(self.conv4(x))
        x = self.pool(f.relu(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        output = f.log_softmax(x, dim=1)
        return output


def training(num_epochs, model, data_batch, valid_batch, device):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sch = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    train_losses = []
    val_losses = []
    acc_val =[]
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(data_batch, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        sch.step()
        loss_val = train_loss / len(data_batch)
        train_losses.append(loss_val)
        val_loss, val_acc = evaluate(model, valid_batch, device)
        val_losses.append(val_loss)
        acc_val.append(val_acc)
        curr_lr = sch.get_last_lr()[0]
        print(f"Current Learning Rate:{curr_lr}")
        print(f"Epoch{epoch+1}: Train loss:{loss_val:.4f} Validation Loss:{val_loss:.4f} Accuracy:{val_acc:.2f}%")
        plot_metrics(range(1, num_epochs+1), train_losses, val_losses, acc_val)

def evaluate(model, data_batch, device):
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_batch:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            acc += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc_rate = 100 * correct / total
    avg_acc = acc/len(data_batch)
    return avg_acc, acc_rate

def plot_metrics(epochs, train_losses, val_losses, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_data, validation_data, testing_data = data_handling()
    cnn_network = CNN().to(device)
    training(60, cnn_network, train_data, validation_data, device)
    test_loss, test_acc = evaluate(cnn_network, testing_data, device)
    print(f"Test Loss:{test_loss:.4f} Test Accuracy: {test_acc:.2f}%")

