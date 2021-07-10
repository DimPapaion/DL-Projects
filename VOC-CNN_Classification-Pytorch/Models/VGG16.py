import torch
import torch.nn as nn


class MyVGG16Model(nn.Module):
    def __init__(self, channels, classes, momentum):
        super(MyVGG16Model, self).__init__()
        self.channels = channels
        self.momentum = momentum

        # Creating the Convolution Layers
        self.Conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fully_Con_Layers = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, classes),

        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully_Con_Layers(x)
        return x
