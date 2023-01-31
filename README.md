
# Semi-supervised Cloud Detection (SSCDnet) in Satellite Images by Considering Domain Shift Problem


For semi-supervised cloud detection, we take domain shift problem into account the semi-supervised learning (SSL) network. Feature-level and output-level domain adaptations are applied to reduce the domain distribution gaps between labeled and unlabeled images, thus improving predicted results accuracy of the SSL network.
Experimental results on Landsat-8 OLI (https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data) and GF-1 WFV (http://sendimage.whu.edu.cn/en/mfc-validation-data/) multispectral images demonstrate that the proposed semi-supervised cloud detection network (SSCDnet) is able to achieve promising cloud detection performance when using a limited number of labeled samples and outperforms several state-of-the-art SSL methods
![](framework.png)

## Package pre-requisites
The code runs on Python 3 and Pytorch 0.4 The following packages are required. 

```
pip install scipy tqdm matplotlib numpy opencv-python
apt-get update -y
apt-get install libglib2.0-0
```

## Dataset preparation

Download ImageNet pretrained Resnet-101([Link](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)) and place it ```./pretrained_models/```


## Training and Validation

### Training and Validation 
```
python train_SSCDnet.py   python evaluate.py 
```
##  Limitations
Although SSCDnet shows good performance, there is still much room for improvement, such as hyper-parameters setting of loss function and threshold setting of pseudolabeling. Different cloud detection datasets have different domain distributions. You may need to update these parameters to achieve a promising performance on different datasets. In addition, different ground objects have different characteristics, and the performance of SSCDnet on other objects detection also needs to be further evaluated.

## Instructions for setting-up Multi-Label Mean-Teacher branch
This work is based on the [Semi-supervised Semantic Segmentation with High- and Low-level Consistency](https://arxiv.org/pdf/1908.05724.pdf).
code available:
https://github.com/sud0301/semisup-semseg

## Acknowledgement

Parts of the code have been adapted from: 
[DeepLab-Resnet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab), [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
[mean-teacher](https://github.com/CuriousAI/mean-teacher) 

## Citation
This paper has been published by Remote sensing.

MDPI and ACS Style

Guo, J.; Xu, Q.; Zeng, Y.; Liu, Z.; Zhu, X. Semi-Supervised Cloud Detection in Satellite Images by Considering the Domain Shift Problem. Remote Sens. 2022, 14, 2641. https://doi.org/10.3390/rs14112641

AMA Style

Guo J, Xu Q, Zeng Y, Liu Z, Zhu X. Semi-Supervised Cloud Detection in Satellite Images by Considering the Domain Shift Problem. Remote Sensing. 2022; 14(11):2641. https://doi.org/10.3390/rs14112641

Chicago/Turabian Style

Guo, Jianhua, Qingsong Xu, Yue Zeng, Zhiheng Liu, and Xiaoxiang Zhu. 2022. "Semi-Supervised Cloud Detection in Satellite Images by Considering the Domain Shift Problem" Remote Sensing 14, no. 11: 2641. https://doi.org/10.3390/rs14112641

Bib Tex:
##


    @article{Guo2022semi,
      title={Semi-Supervised Cloud Detection in Satellite Images by Considering the Domain Shift Problem},
      author={Guo, Jianhua and Xu, Qingsong and Zeng, Yue and Liu, Zhiheng and Zhu, Xiaoxiang},
      journal={Remote Sensing},
      year={2022},
      volume = {14},
      number = {11},
      article-number = {2641},
      url = {https://www.mdpi.com/2072-4292/14/11/2641},
      ISSN = {2072-4292},
      doi = {10.3390/rs14112641}
    }

