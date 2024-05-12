import numpy as np
import sys

# sys.path.append("models/gtn")
# from transformer import GTN

sys.path.append("models/gtn")
from gtn_modified import GTN_Modified

# sys.path.append("models/eeg_net1")
# from eeg_net1 import EEGNet as EEGNet1

sys.path.append("models/eeg_net")
from eeg_net import EEGNet as EEGNet

sys.path.append("models/deep_conv_net")
from deep_conv_net import DeepConvNet

sys.path.append("models/shallow_conv_net")
from shallow_conv_net import ShallowConvNet

sys.path.append("models/fbcnet")
from fbcnet import FBCNet

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_model_class(model_name):
    if model_name == "gtn":
        return GTN_Modified
    elif model_name == "eeg_net":
        return EEGNet
    elif model_name == "deep_conv_net":
        return DeepConvNet
    elif model_name == "shallow_conv_net":
        return ShallowConvNet
    elif model_name == "fbcnet":
        return FBCNet
    else:
        raise ValueError("Invalid model name")