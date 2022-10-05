import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import multi_classes
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
from pytorch_lightning.metrics import ConfusionMatrix
import os 
import cv2

os.makedirs('debug/',exist_ok=True)

def get_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/',type=str, help='the path of dataset')
    parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs(f'debug/',exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_files = list(df.image_id)
    test_dataset = multi_classes(args.dataset,test_files, get_transform())
    model = torch.load(args.model)
    model = model.cuda()
    sfx = nn.Softmax(dim=1)
    cfs = ConfusionMatrix(3)
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    since = time.time()
    
    for image_id in test_files:
        img = cv2.imread(f'data/{image_id}/images/{image_id}.png')
        img = cv2.resize(img, ((256,256)))
        cv2.imwrite(f'debug/{image_id}.png',img)

    with torch.no_grad():
        for img, mask, mask2, img_id in test_dataset:
            
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)
            
            ts_sfx = sfx(pred)
            pred = sfx(pred)
            img_class = torch.max(ts_sfx,1).indices.cpu()
            pred = torch.max(pred,1).indices.cpu()
            mask = torch.max(mask,1).indices.cpu()
            mask_draw = mask.clone().detach()
            
            if args.debug:
        
                img_numpy = pred.detach().numpy()[0]
                img_numpy[img_numpy==1] = 127
                img_numpy[img_numpy==2] = 255
                cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy)
                
                mask_numpy = mask_draw.detach().numpy()[0]
                mask_numpy[mask_numpy==1] = 127
                mask_numpy[mask_numpy==2] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png',mask_numpy)
               
            cfsmat = cfs(img_class,mask).numpy()
            
            sum_iou = 0
            sum_prec = 0
            sum_acc = 0
            sum_recall = 0
            sum_f1 = 0
            
            for i in range(3):
                tp = cfsmat[i,i]
                fp = np.sum(cfsmat[0:3,i]) - tp
                fn = np.sum(cfsmat[i,0:3]) - tp
                
              
                tmp_iou = tp / (fp + fn + tp)
                tmp_prec = tp / (fp + tp + 1) 
                tmp_acc = tp
                tmp_recall = tp / (tp + fn)
                
                
                sum_iou += tmp_iou
                sum_prec += tmp_prec
                sum_acc += tmp_acc
                sum_recall += tmp_recall
                
            
            sum_acc /= (np.sum(cfsmat)) 
            sum_prec /= 3
            sum_recall /= 3
            sum_iou /= 3
            sum_f1 = 2 * sum_prec * sum_recall / (sum_prec + sum_recall)
            
            iou_score.append(sum_iou)
            acc_score.append(sum_acc)
            pre_score.append(sum_prec)
            recall_score.append(sum_recall)
            f1_score.append(sum_f1)
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',np.mean(iou_score),np.std(iou_score))
    print('mean accuracy:',np.mean(acc_score),np.std(acc_score))
    print('mean precsion:',np.mean(pre_score),np.std(pre_score))
    print('mean recall:',np.mean(recall_score),np.std(recall_score))
    print('mean F1-score:',np.mean(f1_score),np.std(f1_score))
