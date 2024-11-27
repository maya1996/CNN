
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as f


#data handling, the data is downloaded and test data is spilted to perform training and validating
def data_handling():

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def training(num_epochs, model, data_batch):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    train_losses = []
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
            loss_val = train_loss / len(data_batch)
            train_losses.append(loss_val)

        print(f"Epoch{epoch+1}: Train loss:{loss_val}")

def evaluate(model, data_batch):
    correct = 0
    acc = 0
    with torch.no_grad():
        for data in data_batch:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc_rate = correct / len(data_batch)

    print(f"Accuracy rate: {acc_rate}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, validation_data, testing_data = data_handling()
    cnn_network = CNN().to(device)
    training(30, cnn_network, train_data)
    evaluate(cnn_network, testing_data)


