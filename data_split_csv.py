import pandas as pd
import numpy as np
import os
import argparse


def pre_csv(data_path,frac):
    np.random.seed(42)
    image_ids = os.listdir(data_path)
    data_size = len(image_ids)
    train_size = int(round(len(image_ids) * frac, 0))
    train_set = np.random.choice(image_ids,train_size,replace=False)
    ds_split = []
    for img_id in image_ids:
        if img_id in train_set:
            ds_split.append('train')
        else:
            ds_split.append('test')
    
    ds_dict = {'image_id':image_ids,
               'category':ds_split 
        }
    df = pd.DataFrame(ds_dict)
    df.to_csv('src/test_train_data.csv',index=False)
    print('Number of train sample: {}'.format(len(train_set)))
    print('Number of test sample: {}'.format(data_size-train_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--dataset', type=str, default='../datasets/DSB2018/image', help='the path of images') # issue 16
    parser.add_argument('--size', type=float, default=0.9, help='the size of your train set')
    args = parser.parse_args()
    os.makedirs('src/',exist_ok=True)
    pre_csv(args.dataset,args.size)
