import res_net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np


def clusterize(weights_tensor, num_clusters):

    w_shape = weights_tensor.size()
    straighted = weights_tensor.view(-1, w_shape[-1] * w_shape[-1])

    cluster_ids_x, cluster_centers = kmeans(X=straighted, num_clusters=num_clusters)
    return torch.nn.Parameter(cluster_centers[cluster_ids_x].view(w_shape))


def main():

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

    net = ResNet()
    net.load_state_dict(torch.load('./cifar_net_my_100_epoch_001lr.pth', map_location=torch.device('cpu')))
    net.eval()

    with torch.no_grad():
        for name, W in net.named_parameters():
            if ('weight' in name)&(W[1].dim()>1):
                new_weights = clusterize(W[1], 3)
                W[1] = new_weights



    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()
    
