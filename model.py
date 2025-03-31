import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res.fc = nn.Linear(self.res.fc.in_features, 10)

    def forward(self, x):
        return self.res(x)