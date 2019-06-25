# encoding=utf-8

'''
torch 学习记录


'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np


cuda = torch.cuda.is_available()


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./picture',
                                        train=True, download=True, transform=transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./picture',
                                       train=False,download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset,batch_size=4, shuffle=True,num_workers=2)


class_name = (
    'plane','car','bird','cat','deer','dog','frog','horse','ship','truck'
)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()

if cuda is False:
    imshow((torchvision.utils.make_grid(images)))
    plt.show()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)


for e in range(2):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        if cuda:
            inputs = inputs.cuda()
            labels = inputs.cuda()

        optimizer.zero_grad()

        outputs = net.cuda(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().numpy()
        if i % 2000 == 0:
            print('%d,%d loss: %0.3f'%(e, i , running_loss/ 2000))
            running_loss = 0

print('finished trainning')


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicted == labels).sum()


print('acc score is {} '.format(correct / total))