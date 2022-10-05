from pytorch_dcsaunet import DCSAU_Net
from thop import profile
import torch

model = DCSAU_Net.Model()


randn_input = torch.randn(1, 3, 256, 256) 
flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
