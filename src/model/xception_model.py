import torch
import torch.nn as nn
import timm  # this is the key change

class XceptionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        
        # Load pretrained xception model from timm
        self.model = timm.create_model('xception', pretrained=True)
        
        # Replace final layer to match num_classes
        in_features = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        return self.model(x)
