import argparse
import os
import sys
import random
import timeit

import cv2
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform


from deeplabv2_unsupervised import Deeplab_maxmin_model,Deeplab_maxmin_model_all, entropy_discriminator, Split_feature_discriminator_0, Split_feature_discriminator_4, Split_feature_discriminator_2
#from model.deeplabv3p import Res_Deeplab 

from discriminator import s4GAN_discriminator
from loss import CrossEntropy2d
from data.voc_dataset import VOCDataSet, VOCGTDataSet
from data import get_loader, get_data_path
from data.augmentations import *

start = timeit.default_timer()


DATA_DIRECTORY = '/home/Datasets/Landsat8_dataset/'
DATA_LIST_PATH = '/home/Datasets/Landsat8_dataset/percentage/10/train_landsat8.txt' # labeld 
DATA_LIST_PATH2 = '/home/Datasets/Landsat8_dataset/percentage/10/train_landsat8_remain.txt'  # unlabeled




CHECKPOINT_DIR = './checkpoints_class_alignment/SSCDnet/'


GPU_NUMBER=0
os.environ['CUDA_VISIBLE_DEVICES']='3'

#IMG_MEAN = np.array((62.00698793,58.66876762,58.67891434 ), dtype=np.float32) # gf
IMG_MEAN = np.array((69.44698793,51.68876762,49.67891434), dtype=np.float32) # landsat
NUM_CLASSES = 2 # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes 
DATASET = 'pascal_voc' #pascal_voc or pascal_context 

SPLIT_ID = './splits/voc/split_0.pkl'

MODEL = 'DeepLab'
BATCH_SIZE = 4
NUM_STEPS = 50000
SAVE_PRED_EVERY = 5000

START_ITER =0


INPUT_SIZE = '321,321'
IGNORE_LABEL = 255 # 255 for PASCAL-VOC / -1 for PASCAL-Context / 250 for Cityscapes

RESTORE_FROM = '/home/Semisupervised_cloud_maskpretrain/pretrained_models/resnet101-5d3b4d8f.pth'

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_WORKERS = 4
RANDOM_SEED = 1234


LAMBDA_ALIGN=0.001
LAMBDA_FM = 0.01
LAMBDA_ST = 1.0
THRESHOLD_ST = 0.6 # 0.6 for PASCAL-VOC/Context / 0.7 for Cityscapes
THRESHOLD_VALUE= 0.55
LABELED_RATIO = None  #0.02 # 1/8 labeled data by default

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpu", type=int, default=GPU_NUMBER,
                        help="choose gpu device.")	
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
						
    parser.add_argument("--data-list2", type=str, default=DATA_LIST_PATH2,
                        help="Path to the file listing the images in the dataset.")						
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
    parser.add_argument("--lambda-align", type=float, default=LAMBDA_ALIGN,
                        help="lambda_st for self-training.")					
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset to be used")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")

    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of the labeled data to full dataset")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
						
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    parser.add_argument("--threshold-value", type=float, default=THRESHOLD_VALUE,
                        help="threshold_value for the self-training threshold.")						
						
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D2", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--restore-from-D31", type=str, default=None,
                        help="Where restore model parameters from.")	
    parser.add_argument("--start-iter", type=str, default=START_ITER,
                        help="Path to the file listing the images in the dataset.")		
    parser.add_argument("--restore-from-D41", type=str, default=None,
                        help="Where restore model parameters from.")

    parser.add_argument("--restore-from-D_D31", type=str, default=None,
                        help="Where restore model parameters from.")
	
    parser.add_argument("--restore-from-D_D41", type=str, default=None,
                        help="Where restore model parameters from.")							
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")

    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")

    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda()  # Ignore label ??
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()  # N,H,W
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)  # N,C,H,W
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy() #  c,H,W
    output = output.transpose((1,2,0))  # H,W,c
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int) # H,W; obtain the index thatrepresented the max value through the axis==2 (i.e., channel)
    output = torch.from_numpy(output).float()  # numpy-->torch-->torch float 
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # get the pred_all[*] map large than threshold value 
                label_sel[num_sel] = compute_argmax_map(pred_all[j]) # score map --> label map with channel==1

                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 
		
		
def find_good_maps_not1t2(D_outs, pred_all):
    count = 0
    # for i in range(D_outs.size(0)):  # N,C
        # if D_outs[i] > args.threshold_st:
            # count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            #if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # get the pred_all[*] map large than threshold value 
                label_sel[num_sel] = compute_argmax_map(pred_all[j]) # score map --> label map with channel==1

                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 		
		
		
		

		
def compute_ignore_mask(pred0, max_pred):
    pred0 = pred0.detach() # c,H,W    
    pred = torch.chunk(torch.squeeze(pred0,0),2,dim=0)
    pred_1 = torch.squeeze(pred[0],0)	# 1,h,w-->h,w
    pred_1 = pred_1.cpu().numpy() 
    pred_1[pred_1 > args.threshold_value] = 0
    pred_1[pred_1 < 1-args.threshold_value] = 0
    pred_1[pred_1 > 0] = 255    #h,w
    max_pred = max_pred.cpu().numpy() 	
    mask = 	max_pred + pred_1
    mask[mask > 2] = 255  	
    mask =torch.from_numpy(mask) #h,w
    
    return mask	


def find_good_maps_new(D_outs, pred_all, pred_all_2):
    count = 0
    for i in range(D_outs.size(0)):  # N,C
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        #print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)) # n,c,h,w
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)) # n,h,w
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]  # c,h,w; get the pred_all[*] map large than threshold value 
                #label_sel[num_sel] = compute_argmax_map(pred_all[j]) # H,W; score map --> label map with channel==1
                label_sel[num_sel] = compute_ignore_mask( pred_all_2[j], compute_argmax_map(pred_all[j]) )
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 
				



criterion = nn.BCELoss()

def main():
    print (args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Deeplab_maxmin_model_all(num_classes=args.num_classes)
    
    # load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)
    
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
	
    #model.load_state_dict(torch.load('./checkpoints_ablation/voc_s4GAN_threshold_ab_fm1st1_t1t2_5/VOC_35000.pth'))	
    
    model.train()
    model.cuda()

    cudnn.benchmark = True

    # init D
    model_D = s4GAN_discriminator(num_classes=args.num_classes, dataset=args.dataset)

    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
		
    #model_D.load_state_dict(torch.load('./checkpoints_ablation/voc_s4GAN_threshold_ab_fm1st1_t1t2_5/VOC_35000_D.pth'))		
    model_D.train()
    model_D.cuda()
	
	
	
    ###################################################################	
    # init D31	
    model_D31 = Split_feature_discriminator_0(layer_name='layer3')
    if args.restore_from_D31 is not None:
        model_D31.load_state_dict(torch.load(args.restore_from_D31))
    #model_D31.load_state_dict(torch.load('./checkpoints_advent_feature/voc_advent_feature_landsat_layer3/VOC_60000_D31.pth'))	
    model_D31.train()
    model_D31.cuda()		
	
	#####################################################################
    # init D41	
    model_D41 = Split_feature_discriminator_0(layer_name='layer4')
    if args.restore_from_D41 is not None:
        model_D41.load_state_dict(torch.load(args.restore_from_D41))
    #model_D41.load_state_dict(torch.load('./checkpoints_advent_feature/voc_advent_feature_landsat_layer3/VOC_60000_D41.pth'))	
    model_D41.train()
    model_D41.cuda()	

    #################################################################################################

	
    # init D_D31	
    model_D_D31 = Split_feature_discriminator_0(layer_name='layer3')
    if args.restore_from_D_D31 is not None:
        model_D_D31.load_state_dict(torch.load(args.restore_from_D_D31))
    #model_D_D31.load_state_dict(torch.load('./checkpoints_advent_feature/voc_advent_feature_landsat_layer3/VOC_60000_D_D31.pth'))	
    model_D_D31.train()
    model_D_D31.cuda()	

	#####################################################################
    # init D_D41	
    model_D_D41 = Split_feature_discriminator_0(layer_name='layer4')
    if args.restore_from_D_D41 is not None:
        model_D_D41.load_state_dict(torch.load(args.restore_from_D_D41))
    #model_D_D41.load_state_dict(torch.load('./checkpoints_advent_feature/voc_advent_feature_landsat_layer3/VOC_60000_D_D41.pth'))	
    model_D_D41.train()
    model_D_D41.cuda()		
	
	

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # load data and do preprocessing,such as rescale,flip
    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    train_remain_dataset = VOCDataSet(args.data_dir, args.data_list2, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)						


    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
    trainloader_remain = data.DataLoader(train_remain_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    trainloader_remain_iter = iter(trainloader_remain)

    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)
    
    

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()
	
	
    optimizer_D31 = optim.Adam(model_D31.parameters(), lr=1*args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D31.zero_grad()		


    optimizer_D41 = optim.Adam(model_D41.parameters(), lr=1*args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D41.zero_grad()		

	
    optimizer_D_D31 = optim.Adam(model_D_D31.parameters(), lr=1*args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D_D31.zero_grad()		


    optimizer_D_D41 = optim.Adam(model_D_D41.parameters(), lr=1*args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D_D41.zero_grad()			

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())


    for i_iter in range(args.num_steps):
    #for i_iter in  range(35001, args.num_steps+1):    
        loss_ce_value = 0
        loss_D_value = 0
        loss_D2_value = 0		
		
        loss_fm_value = 0
        loss_S_value = 0
        loss_st_value = 0
		
        loss_D31_value = 0	
        loss_D41_value = 0	
        loss_D_D31_value = 0	
        loss_D_D41_value = 0	
		

        loss_adv_trg_value =0
        loss_align_value = 0		
		
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
		
		
        optimizer_D31.zero_grad()
        adjust_learning_rate_D(optimizer_D31, i_iter)

        optimizer_D41.zero_grad()
        adjust_learning_rate_D(optimizer_D41, i_iter)

        optimizer_D_D31.zero_grad()
        adjust_learning_rate_D(optimizer_D_D31, i_iter)

        optimizer_D_D41.zero_grad()
        adjust_learning_rate_D(optimizer_D_D41, i_iter)				
		

        # train Segmentation Network 
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
			
			
			
        for param in model_D31.parameters():
            param.requires_grad = False			

        for param in model_D41.parameters():
            param.requires_grad = False			
		
        for param in model_D_D31.parameters():
            param.requires_grad = False				
			
        for param in model_D_D41.parameters():
            param.requires_grad = False					

        ########################## 1. training loss for labeled data only  #############################
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = Variable(images).cuda()
        pred, _, _, feature_img3, feature_img4 = model(images)
		
		
		###############################################################################				
        pred_softmax = F.softmax(pred, dim=1)
        mask_pred= torch.chunk(pred_softmax,2,dim=1) # split, [N,C,H,W]-->[N,C/2,H,W]
		
		# class-relevrant features selection
        y31 = torch.mul(mask_pred[0],feature_img3)		
        D31_out_y, D31_out_y_pred = model_D31(y31) 
		
        x31 = torch.mul(mask_pred[1],feature_img3)		
        D31_out_x, D31_out_x_pred = model_D_D31(x31) 

        y41 = torch.mul(mask_pred[0],feature_img4)
        D41_out_y, D41_out_y_pred = model_D41(y41) 
		
        x41 = torch.mul(mask_pred[1],feature_img4)
        D41_out_x, D41_out_x_pred = model_D_D41(x41) 			
		
				
        pred = interp(pred)		
        loss_ce = loss_calc(pred, labels) # Cross entropy loss for labeled data
		
		

        ############################ 2. training loss for remaining unlabeled data  ####################
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain, _, _, _, _ = batch_remain
        images_remain = Variable(images_remain).cuda()
        pred_remain, _, _, feature_remain_img3, feature_remain_img4 = model(images_remain)
		
		
		
		###############################################################################				
        pred_remain_softmax = F.softmax(pred_remain, dim=1)
        mask_pred_remain= torch.chunk(pred_remain_softmax,2,dim=1) # split, [N,C,H,W]-->[N,C/2,H,W]		
		
		# class-relevrant features selection
        yr31 = torch.mul(mask_pred_remain[0],feature_remain_img3)		
        D31_out_yr_remain, D31_out_yr_pred_remain = model_D31(yr31) 
		
        xr31 = torch.mul(mask_pred_remain[1],feature_remain_img3)		
        D31_out_xr_remain, D31_out_xr_pred_remain = model_D_D31(xr31) 

        yr41 = torch.mul(mask_pred_remain[0],feature_remain_img4)
        D41_out_yr_remain, D41_out_yr_pred_remain = model_D41(yr41) 
		
        xr41 = torch.mul(mask_pred_remain[1],feature_remain_img4)
        D41_out_xr_remain, D41_out_xr_pred_remain = model_D_D41(xr41) 			
				
		
		
		###############################################################################		
        pred_remain	= interp(pred_remain)
        images_remain = (images_remain-torch.min(images_remain))/(torch.max(images_remain)- torch.min(images_remain))
        #print (pred_remain.size(), images_remain.size())
		
        #pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain), dim=1) # un-modify

        pred_remain_2 = F.softmax(pred_remain, dim=1)
        mask1 = torch.chunk(pred_remain_2,2,dim=1)
        mask_11 = torch.cat([mask1[0], mask1[0], mask1[0]], dim=1)
        mask_12 = torch.cat([mask1[1], mask1[1], mask1[1]], dim=1)		
        pred_cat = torch.cat( [ images_remain.mul(mask_11.type(torch.cuda.FloatTensor)), images_remain.mul(mask_12.type(torch.cuda.FloatTensor)) ], dim=1 ) 
        ###############################################################################
          
        D_out_z, D_out_y_pred = model_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss 
  
        # find predicted segmentation maps above threshold 
        pred_sel, labels_sel, count = find_good_maps_new(D_out_z, pred_remain, pred_remain_2) #find_good_maps(D_out_z, pred_remain)   #
			
			
        if count > 0 and i_iter > args.start_iter:
            loss_st = loss_calc(pred_sel, labels_sel)
        else:
            loss_st = 0.0

        ################ 4. Concatenates the input images and ground-truth maps for the Districrimator 'Real' input ###############
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt, _, _, _ = batch_gt
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        D_gt_v = Variable(one_hot(labels_gt)).cuda()
		
        images_gt = images_gt.cuda()
        images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))
        ###############################################################################
        mask2 = torch.chunk(D_gt_v,2,dim=1)		
        mask_21 = torch.cat([mask2[0], mask2[0], mask2[0]], dim=1)
        mask_22 = torch.cat([mask2[1], mask2[1], mask2[1]], dim=1)		
        D_gt_v_cat = torch.cat( ( images_gt.mul(mask_21.type(torch.cuda.FloatTensor)), images_gt.mul(mask_22.type(torch.cuda.FloatTensor)) ), dim=1 )		
		
        ###############################################################################  
		
        #D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1) # un-modify
        D_out_z_gt , D_out_y_gt = model_D(D_gt_v_cat)
        
        # L1 loss for Feature Matching Loss
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
		
		
		
		# feature-level alignment loss FA
		
        loss_fa_layer3 = torch.mean(torch.abs(torch.mean(D31_out_yr_pred_remain, 0) - torch.mean(D31_out_y_pred, 0))) + torch.mean(torch.abs(torch.mean(D31_out_xr_pred_remain, 0) - torch.mean(D31_out_x_pred, 0)))
		
        loss_fa_layer4 = torch.mean(torch.abs(torch.mean(D41_out_yr_pred_remain, 0) - torch.mean(D41_out_y_pred, 0))) + torch.mean(torch.abs(torch.mean(D41_out_xr_pred_remain, 0) - torch.mean(D41_out_x_pred, 0)))
		
        loss_align =loss_fa_layer3+ loss_fa_layer4			
		
		
        loss_S = loss_ce +  args.lambda_fm*loss_fm + args.lambda_st*loss_st  + args.lambda_align*loss_align		
    
        # if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
            # loss_S = loss_ce +  args.lambda_fm*loss_fm + args.lambda_st*loss_st  + args.lambda_align*loss_align
        # else:
            # loss_S = loss_ce + args.lambda_fm*loss_fm

        loss_S.backward()
        loss_fm_value+= 1*loss_fm 
        loss_st_value += loss_st
        loss_ce_value += loss_ce.item()
        loss_align_value +=1*loss_align		
        loss_S_value += loss_S.item()

        ###################################################### 5.train D  #################################################
        for param in model_D.parameters():
            param.requires_grad = True

        # train with pred
        pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.
        
        D_out_z, _ = model_D(pred_cat)
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_z, y_fake_) 

        # train with gt
        D_out_z_gt , _ = model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_z_gt, y_real_)
        
        loss_D = (loss_D_fake + loss_D_real)/2.0
        loss_D.backward()
        loss_D_value += loss_D.item()
		
		
        ###################################################### 6.train D31  #################################################
        for param in model_D31.parameters():
            param.requires_grad = True

        # train with pred
        feature_remain_img = y31.detach()  # detach does not allow the graddients to back propagate.
        
        D31_out_z3_remain, _ = model_D31(feature_remain_img) 	
        #y3_fake_ = Variable(torch.zeros(D_out_z3.size(0), 1).cuda(args.gpu))
        y3_fake_ = Variable(torch.zeros(D31_out_z3_remain.size(0), 1).cuda())		
        loss_D31_fake = criterion(D31_out_z3_remain, y3_fake_) 

        # train with gt
        feature_img = yr31.detach()		
        D31_out_z3, _ = model_D31(feature_img)
        #y3_real_ = Variable(torch.ones(D_out_z3_gt.size(0), 1).cuda(args.gpu)) 
        y3_real_ = Variable(torch.ones(D31_out_z3.size(0), 1).cuda()) 		
        loss_D31_real = criterion(D31_out_z3, y3_real_)
        
        loss_D31 = (loss_D31_fake + loss_D31_real)/2.0
        loss_D31.backward()
        loss_D31_value += loss_D31.item()	

        ###################################################### 6.train D41  #################################################
        for param in model_D41.parameters():
            param.requires_grad = True

        # train with pred
        feature_remain_img = y41.detach()  # detach does not allow the graddients to back propagate.
        
        D41_out_z4_remain, _ = model_D41(feature_remain_img) 	
        #y4_fake_ = Variable(torch.zeros(D_out_z4.size(0), 1).cuda(args.gpu))
        y4_fake_ = Variable(torch.zeros(D41_out_z4_remain.size(0), 1).cuda())		
        loss_D41_fake = criterion(D41_out_z4_remain, y4_fake_) 

        # train with gt
        feature_img = yr41.detach()		
        D41_out_z4, _ = model_D41(feature_img)
        #y4_real_ = Variable(torch.ones(D_out_z4_gt.size(0), 1).cuda(args.gpu)) 
        y4_real_ = Variable(torch.ones(D41_out_z4.size(0), 1).cuda()) 		
        loss_D41_real = criterion(D41_out_z4, y4_real_)
        
        loss_D41 = (loss_D41_fake + loss_D41_real)/2.0
        loss_D41.backward()
        loss_D41_value += loss_D41.item()		
		
		
       ###################################################### 6.train D_D31  #################################################
        for param in model_D_D31.parameters():
            param.requires_grad = True

        # train with pred
        feature_remain_img = x31.detach()  # detach does not allow the graddients to back propagate.
        
        D31_out_z3_remain, _ = model_D_D31(feature_remain_img) 	
        #y3_fake_ = Variable(torch.zeros(D_out_z3.size(0), 1).cuda(args.gpu))
        y3_fake_ = Variable(torch.zeros(D31_out_z3_remain.size(0), 1).cuda())		
        loss_D_D31_fake = criterion(D31_out_z3_remain, y3_fake_) 

        # train with gt
        feature_img = xr31.detach()		
        D31_out_z3, _ = model_D_D31(feature_img)
        #y3_real_ = Variable(torch.ones(D_out_z3_gt.size(0), 1).cuda(args.gpu)) 
        y3_real_ = Variable(torch.ones(D31_out_z3.size(0), 1).cuda()) 		
        loss_D_D31_real = criterion(D31_out_z3, y3_real_)
        
        loss_D_D31 = (loss_D_D31_fake + loss_D_D31_real)/2.0
        loss_D_D31.backward()
        loss_D_D31_value += loss_D_D31.item()	


        ###################################################### 6.train D_D41  #################################################
        for param in model_D_D41.parameters():
            param.requires_grad = True

        # train with pred
        feature_remain_img = x41.detach()  # detach does not allow the graddients to back propagate.
        
        D41_out_z4_remain, _ = model_D_D41(feature_remain_img) 	
        #y4_fake_ = Variable(torch.zeros(D_out_z4.size(0), 1).cuda(args.gpu))
        y4_fake_ = Variable(torch.zeros(D41_out_z4_remain.size(0), 1).cuda())		
        loss_D_D41_fake = criterion(D41_out_z4_remain, y4_fake_) 

        # train with gt
        feature_img = xr41.detach()		
        D41_out_z4, _ = model_D_D41(feature_img)
        #y4_real_ = Variable(torch.ones(D_out_z4_gt.size(0), 1).cuda(args.gpu)) 
        y4_real_ = Variable(torch.ones(D41_out_z4.size(0), 1).cuda()) 		
        loss_D_D41_real = criterion(D41_out_z4, y4_real_)
        
        loss_D_D41 = (loss_D_D41_fake + loss_D_D41_real)/2.0
        loss_D_D41.backward()
        loss_D_D41_value += loss_D_D41.item()	

		
		

        optimizer.step()
        optimizer_D.step()
        optimizer_D31.step()	
        optimizer_D41.step()
        optimizer_D_D31.step()
        optimizer_D_D41.step()
		
		
		
        if i_iter %20 ==0:
            print('iter={0:5d}, loss_S={1:.3f}, loss_fm={2:.3f}, loss_align={3:.3f}, loss_st={4:.3f}, loss_ce={5:.3f}'.format(i_iter, loss_S_value, loss_fm_value, loss_align_value, loss_st_value, loss_ce_value))

            print('loss_D={0:.3f}, loss_D31={1:.3f}, loss_D41={2:.3f}, loss_D_D31={3:.3f}, loss_D_D41={4:.3f}'.format(loss_D_value, loss_D31_value, loss_D41_value, loss_D_D31_value, loss_D_D41_value))
			
			
        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))		
            torch.save(model_D31.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D31.pth'))			
            torch.save(model_D41.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D41.pth'))
            torch.save(model_D_D31.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D_D31.pth'))			
            torch.save(model_D_D41.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(args.num_steps)+'_D_D41.pth'))			
			
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint  ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D.pth'))		
            torch.save(model_D31.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D31.pth'))			
            torch.save(model_D41.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D41.pth'))
            torch.save(model_D_D31.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D_D31.pth'))			
            torch.save(model_D_D41.state_dict(),os.path.join(args.checkpoint_dir, 'VOC_'+str(i_iter)+'_D_D41.pth'))
			
			
			
    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
