import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np


class BaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        self.subsampling = 2 if in_channels != out_channels else 1
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.subsampling, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.sequential(x)
        return out

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, base_block = BaseBlock):

        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_block = base_block(in_channels, out_channels)
        self.block_expantion = self.base_block.subsampling
        self.shortcut = nn.Identity()
        if self.base_block.subsampling==2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=self.block_expantion,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )



    def forward(self, x):

        residual = self.shortcut(x)
        out = self.base_block(x)
        out += residual
        out = F.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResBlock, n=1):
        super(ResLayer, self).__init__()
        self.layer = nn.Sequential(
            block(in_channels, out_channels),
                *[block(out_channels, out_channels) for _ in range(n-1)]
            )
    def forward(self, x):
        out = self.layer(x)
        return out

class ResEnter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEnter, self).__init__()
        self.enter_seq = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        out = self.enter_seq(x)
        return out

class ResExit(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, n_classes, bias=False)
    def forward(self, x):
        out = self.avg(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResMidLine(nn.Module):
    def __init__(self, sizes=[16, 32, 64], depths = [2, 2, 2], block=ResBlock):
        super().__init__()
        self.in_out_block_sizes = list(zip(sizes, sizes[1:]))
        self.blocks = nn.ModuleList([
            ResLayer(sizes[0], sizes[0], n=depths[0], block=block),
            *[ResLayer(in_channels, out_channels, n=n, block=block)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class ResNet(nn.Module):
    def __init__(self, sizes=[16, 32, 64], depths = [2, 2, 2]):
        super(ResNet, self).__init__()
        self.enter = ResEnter(3, sizes[0])
        self.mid = ResMidLine(sizes=sizes, depths = depths)
        self.exit = ResExit(64, 10)
    def forward(self, x):
        out = self.enter(x)
        out = self.mid(out)
        out = self.exit(out)
        return out

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

res_net = ResNet()
res_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = res_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = res_net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = res_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

PATH = './cifar_net_my_100_epoch_001lr.pth'
torch.save(res_net.state_dict(), PATH)
