
# pip install ptflops
# pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
# pip install thop
import torchvision.models as models

import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='1'

from deeplabv2 import Res_Deeplab

from discriminator import s4GAN_discriminator11, FCDiscriminator, s4GAN_discriminator_DAN

from deeplabv2_unsupervised import Split_feature_discriminator_0

import torch
from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
  # net = models.vgg16()
  # macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  

  
with torch.cuda.device(0):
  net = Res_Deeplab() #models.vgg16()
  macs, params = get_model_complexity_info(net, (3, 321, 321), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))



  # net = s4GAN_discriminator_DAN() #models.vgg16()
  # macs, params = get_model_complexity_info(net, (5, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  
  

  # net = FCDiscriminator() #models.vgg16()
  # macs, params = get_model_complexity_info(net, (2, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  
  # net = s4GAN_discriminator11() #models.vgg16()
  # macs, params = get_model_complexity_info(net, (6, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  
  
  # net =Split_feature_discriminator_0(layer_name='layer3')
  
  
  # macs, params = get_model_complexity_info(net, (64*4*4, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  
  
  # net =Split_feature_discriminator_0(layer_name='layer4')
  
  
  # macs, params = get_model_complexity_info(net, (64*4*8, 224, 224), as_strings=True,
                                           # print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))