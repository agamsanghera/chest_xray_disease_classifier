from torchvision import models
import torch
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class XRayClassifier(nn.Module):
    """
    Base CheXNet model, modified to fit constrained classification task.
    State dict must be loaded to initialize model from DenseNet to CheXNet pretrained state.
    ie. use XRayClassifier.load_state_dict(get_chexnet_state(),strict=False)
    """
    def __init__(self, out_size):
        super(XRayClassifier, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.ReLU(),
            nn.Linear(14,out_size), #out_size should be 5 for this application
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
# function to get state dict for the CheXNet Model
def get_chexnet_state(path='./model.pth.tar',map_to=device):
    '''
    Modify given CheXNet state dictionary to fit the modified architecture.
    Returns a state dict which can be passed to XRayClassifier model
    
    NOTE: Set strict=False when passing to load_state_dict method.
    '''
    checkpoint = torch.load(path, map_location=map_to)
    substring = 'module.'
    state = checkpoint['state_dict']
    new_state = {}

    for old_key in state.keys():
        new_key = old_key[len(substring):] if old_key.startswith(substring) else old_key
        new_key = new_key.replace('norm.1', 'norm1')
        new_key = new_key.replace('conv.1', 'conv1')
        new_key = new_key.replace('norm.2', 'norm2')
        new_key = new_key.replace('conv.2', 'conv2')
        new_state[new_key] = state[old_key] 

    return new_state
