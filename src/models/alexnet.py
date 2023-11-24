import torch.nn as nn
from models.base_model import BaseModel
from torchvision.models import AlexNet as _AlexNet
from torchvision.models import alexnet, AlexNet_Weights
import torch

class AlexNet(BaseModel):
    def __init__(self, 
        img_size,
        num_classes=10,
        in_channels = 3,
        dropout=0.5,
        pretrain=False):
        super(AlexNet, self).__init__(img_size)
        """
        AlexNet implementation copied from torch library
        """
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout

        self._activation_func = None # TODO: not implemented, so overwrite baseclass assignment!

        if pretrain:
            loaded_model = alexnet(AlexNet_Weights.DEFAULT)
            if self.num_classes != 1000:
                loaded_model.classifier[-1] = nn.Linear(4096, num_classes)
        else:
            loaded_model = _AlexNet(self.num_classes, self.dropout)
        
        self.features = loaded_model.features
        if self.in_channels != 3:
            self.features[0] = nn.Conv2d(self.in_channels, 64, kernel_size=11, stride=4, padding=2)
        self.avgpool = loaded_model.avgpool
        self.classifier = loaded_model.classifier
        self.classifier.insert(0, nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_sequential(self) -> nn.Module:
        return nn.Sequential(*self.features,self.avgpool, *self.classifier)

    @property
    def classname(self) -> str:
        return "alexnet"
    
    @property
    def _config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "dropout": self.dropout
            }