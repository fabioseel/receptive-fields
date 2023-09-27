import torch
import torch.nn as nn
import yaml
import math

class SimpleCNN(nn.Module):
    def __init__(self, img_size, num_classes, num_layers=3, in_channels=3, num_channels=16,
                 kernel_size=3, stride=1, dilation=1):
        super(SimpleCNN, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size, stride, dilation=dilation)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.extra_conv_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride, dilation=dilation))
        
        # Fully connected layer
        res_size = self.img_size
        for l in range(self.num_layers):
            res_size = math.floor((res_size-dilation*(kernel_size-1)-1)/stride + 1)
        self.fc = nn.Linear(num_channels * res_size**2, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        for conv_layer in self.extra_conv_layers:
            x = conv_layer(x)
            x = self.relu(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        if not self.train:
            x = self.softmax(x)
        return x
    
    def save(self, filename):
        config = {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'in_channels': self.in_channels,
            'num_channels':self.num_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation
        }
        with open(filename+".cfg", 'w') as f:
            yaml.dump(config, f)
        torch.save(self.state_dict(),filename+".pth")

    @classmethod
    def load(cls, filename):
        with open(filename+".cfg", 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        model = cls(**config)
        model.load_state_dict(torch.load(filename+".pth"))
        return model