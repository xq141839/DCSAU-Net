import torch.nn as nn
import torch
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np

cfs = ConfusionMatrix(3)

class DiceLoss_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)
    
    def binary_dice(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1-dice

    def forward(self, ipts, gt):
        
        ipts = self.sfx(ipts)
        c = ipts.shape[1]
        sum_loss = 0
        for i in range(c):
            tmp_inputs = ipts[:,i]
            tmp_gt = gt[:,i]
            tmp_loss = self.binary_dice(tmp_inputs,tmp_gt)
            sum_loss += tmp_loss
        return sum_loss / c
    
class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.sfx(inputs)
        c = inputs.shape[1]
        inputs = torch.max(inputs,1).indices.cpu()
        targets = torch.max(targets,1).indices.cpu()
        cfsmat = cfs(inputs,targets).numpy()
        
        sum_iou = 0
        for i in range(c):
                tp = cfsmat[i,i]
                fp = np.sum(cfsmat[0:3,i]) - tp
                fn = np.sum(cfsmat[i,0:3]) - tp
            
                tmp_iou = tp / (fp + fn + tp)
                sum_iou += tmp_iou
                
        return sum_iou / c

class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1-dice

class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU
