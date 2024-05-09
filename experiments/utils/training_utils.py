import numpy as np
import sys

sys.path.append("models/gtn")
from transformer import GTN

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_model_class(model_name):
    if model_name == "gtn":
        return GTN
    else:
        raise ValueError("Invalid model name")