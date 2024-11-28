
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

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = CIFAR10(root='./data',train=True,download=True,transform=transform_val)
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
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 750)
        self.fc2 = nn.Linear(750, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def training(num_epochs, model, data_batch):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_losses = []
    loss_val = 0
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
            loss_val = train_loss / len(data_batch)
            train_losses.append(loss_val)

        print(f"Epoch{epoch+1}: Train loss:{loss_val}")

def evaluate(model, data_batch):
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_batch:
            images, labels = data
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            acc += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc_rate = 100 * correct / total
    avg_acc = acc/len(data_batch)

    print(f"Test loss: {avg_acc}Accuracy rate: {acc_rate}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, validation_data, testing_data = data_handling()
    cnn_network = CNN().to(device)
    training(30, cnn_network, train_data)
    evaluate(cnn_network, testing_data)


