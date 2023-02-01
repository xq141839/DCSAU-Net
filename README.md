# DCSAU-Net: A Deeper and More Compact Split-Attention U-Net for Medical Image Segmentation
## News
2022.08.25: The DCSAU-Net model has been optimised. The paper will be updated later.

2022.09.27: The updated preprint has been available at [arXiv](https://arxiv.org/pdf/2202.00972v2.pdf). 

2022.10.05: The method of calculating FLOPs, parameters and FPS has been uploaded. 

2022.12.09: A requirements.txt for Linux environment has been uploaded.

2023.02.02: The article has been accepted and available in the journal: [Computers in Biology and Medicine](https://authors.elsevier.com/sd/article/S0010-4825(23)00091-4).
## Requirements
1. pytorch==1.10.0
2. pytorch-lightning==1.1.0
3. albumentations==0.3.2
4. seaborn
5. sklearn
## Dataset
To apply the model on a custom dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
```
## CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```
## Train
```
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 16 --lr 0.001 --epoch 150 
```
## Evaluation
```
python eval_binary.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
## Acknowledgement
The codes are modified from [ResNeSt](https://github.com/zhanghang1989/ResNeSt/tree/5fe47e93bd7e098d15bc278d8ab4812b82b49414), [U-Net](https://github.com/milesial/Pytorch-UNet)
