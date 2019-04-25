from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--img-path', type=str)
    device = torch.device("cpu")
    args = parser.parse_args()
    model = Net()
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    #model = torch.load(args.model_path, map_location='cpu')

    with torch.no_grad():
        im = Image.open(args.img_path)
        tens = transforms.ToTensor()
        imt = tens(im).view(1, -1, im.size[1], im.size[0])
        x = model(imt)
        x = x.cpu()
        x = x.numpy()
        print("The Output is: {}".format(np.argmax(x)))

if __name__ == '__main__':
    main()