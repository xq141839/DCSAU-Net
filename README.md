# DCSAU-Net: A Deeper and More Compact Split-Attention U-Net for Medical Image Segmentation
By [Qing Xu](https://www.linkedin.com/in/%E5%8D%BF-%E5%BE%90-6556a9181/), [Wenting Duan](https://staff.lincoln.ac.uk/wduan) and Na He
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
## Acknowledgement
The codes are modified from [ResNeSt](https://github.com/zhanghang1989/ResNeSt/tree/5fe47e93bd7e098d15bc278d8ab4812b82b49414), [U-Net](https://github.com/milesial/Pytorch-UNet)
