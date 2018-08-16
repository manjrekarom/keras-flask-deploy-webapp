import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        # change filters depending on image sizes
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2) # o.v. 56
        self.conv2 = nn.Conv2d(10, 32, 3, padding=1)
        # newly added
        self.pool2 = nn.MaxPool2d(2, 2) # o.v. 28
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # o.v. 14
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 14 * 14, 1000)
        self.fc2 = nn.Linear(1000, 120)
        self.fc3 = nn.Linear(120, 19) # second parameter is number of classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.conv4(x)

        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = LeNet()
