from model_basic import UnetDa
from init import ModelInit
import numpy as np 
import torch

model_parser = ModelInit()
model = UnetDa(model_parser)

image_sparse = np.random.rand(128,128)
input = torch.from_numpy(image_sparse).unsqueeze_(0).unsqueeze_(0)
out = model(input)

