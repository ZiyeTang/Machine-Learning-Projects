import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from hw2_ResNet import ResNet
import matplotlib.pyplot as plt
import struct
import os
import numpy as np

torch.manual_seed(1)

root = './hw2_data'
if not os.path.exists(root):
    os.mkdir(root)

normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=normalization, download=True)
test_set = dset.MNIST(root=root, train=False, transform=normalization, download=True)
trainLoader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

net = ResNet(64)

numparams = 0
for f in net.parameters():
    print(f.size())
    numparams += f.numel()

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

test_loss = []
train_loss = []

def test(net, testLoader):
    net.eval()
    correct = 0
    tloss = 0
    with torch.no_grad():
        for (data,target) in testLoader:
            output = net(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            tloss += criterion(output, target).item()
        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))
    
    tloss/=len(testLoader)
    test_loss.append(tloss)
    print("test loss:", tloss,"\n")

test(net, testLoader)

for epoch in range(400):
    net.train()
    tloss = 0

    for batch_idx, (data, target) in enumerate(trainLoader):
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        tloss+=loss.item()
        gn = 0
        for f in net.parameters():
            gn = gn + torch.norm(f.grad)
        #print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
        optimizer.step()
        optimizer.zero_grad()

    tloss /= len(trainLoader)
    train_loss.append(tloss)
    print("\ntrain loss:", tloss)
    test(net, testLoader)



a = train_loss
b = test_loss
idx = np.argsort(a)
x = []
y = []
for i in idx:
  x.append(a[i])
  y.append(b[i])

plt.plot(x,y)
plt.xlabel("Train Loss")
plt.ylabel("Test Loss")
