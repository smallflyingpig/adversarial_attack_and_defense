import torch
from torch import nn

class ConvNet(nn.Module):
    # mnist: input shape: (28,28)
    def __init__(self, in_channel=1, n_class=10):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 1, 0, bias=True), #(32,26,26)
            nn.MaxPool2d(2,2,0), #(32,13,13)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3,1,0, bias=True), #(64,11,11)
            nn.MaxPool2d(2,2,0), #(64,5,5)
            nn.ReLU(inplace=True)
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5*5*64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.classifier_layer = \
            nn.Linear(128, n_class)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(B, -1)
        x = self.dense_layers(x)
        x = self.classifier_layer(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

