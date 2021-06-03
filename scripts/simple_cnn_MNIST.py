import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

import sys
sys.path.append('.')
from utils.logger import Logger
from utils import utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.in3 = nn.InstanceNorm2d(16)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.in1(x)
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.in2(x)
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.in3(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
#         x = x/x.norm(dim=(1),keepdim=True)
        x = F.log_softmax(x, dim=1)
        return x
    
def test(model, device, test_loader, epoch, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    logger.add_scalar(epoch, 'test_loss', test_loss)
    logger.add_scalar(epoch, 'test_accuracy', 100. * correct / len(test_loader.dataset))
    logger.add_scalar(epoch, 'correct(of {})'.format(len(test_loader.dataset)), correct)
    
def train_iter(model, data, target, device, optimizer, iter, n_iter, logger):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if 0 == iter % 100:
        logger.print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            iter, iter, n_iter, 100. * iter / n_iter, loss.item()))
    
def main():
    seed = 0
    device = torch.device('cpu')
    n_iter = 2000
    torch.manual_seed(seed)
    
    logger = Logger('simple_cnn_MNIST')
    logger.print('seed={}'.format(seed))
    
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    dataset1 = datasets.MNIST('data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1, batch_size=250, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)
    
    model = Net().to(device)
    logger.print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    data,target = utils.get_random_batch(dataset1, 10)
    for iter in range(n_iter+1):
        train_iter(model, data, target, device, optimizer, iter, n_iter, logger)
        if 0 == iter % 100:
            test(model, device, test_loader, iter, logger)
            logger.iter_info()
            logger.save()
        
    torch.save(model.state_dict(), "./checkpoints/mnist_cnn.pt")

if __name__ == '__main__':
    main()
