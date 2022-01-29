import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import EMINet

import pytorch_ssim
import pytorch_iou
# ------- 1. define loss function --------

#bce_loss = nn.BCELoss(size_average=True)#py0.4
bce_loss = nn.BCELoss(reduction='mean')#py1.4
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df, labels_v, edge_v):

	loss1 = bce_ssim_loss(g1,labels_v)
	loss2 = bce_ssim_loss(g2,labels_v)
	loss3 = bce_ssim_loss(g3,labels_v)
	loss4 = bce_ssim_loss(g4,labels_v)
	loss5 = bce_ssim_loss(g5,labels_v)
	loss6 = bce_ssim_loss(g6,labels_v)
	loss7 = bce_ssim_loss(g7,labels_v)

	losse1 = bce_loss(e1, edge_v)
	losse2 = bce_loss(e2, edge_v)
	losse3 = bce_loss(e3, edge_v)
	losse4 = bce_loss(e4, edge_v)
	losse5 = bce_loss(e5, edge_v)
	losse6 = bce_loss(e6, edge_v)
	
	lossdf = bce_ssim_loss(df,labels_v)

	loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + losse1 + losse2 + losse3 + losse4 + losse5 + losse6 + lossdf

	return lossdf, loss


# ------- 2. set the directory of training dataset --------

tra_img_dir = './Data/trainingDataset/imgs_train_flip/'
tra_lbl_dir = './Data/trainingDataset/masks_train_flip/'
tra_edge_dir = './Data/trainingDataset/edge_train_flip/'

image_ext = '.bmp'
label_ext = '.png'
edge_ext = '.png'

model_dir = './trained_models1/'


epoch_num = 340
batch_size_train = 10
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(tra_img_dir + '*' + image_ext)
tra_lbl_name_list = []
tra_edge_name_list = []

for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]          
        imgIdx = img_name.split(".")[0]
        tra_lbl_name_list.append(tra_lbl_dir + imgIdx + label_ext)
        tra_edge_name_list.append(tra_edge_dir + imgIdx + edge_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train edges: ", len(tra_edge_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    edge_name_list=tra_edge_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True,num_workers=3, pin_memory=True)#num_workers=0

# ------- 3. define model --------
# define the net
net = EMINet(3, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
Loss = []

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, edge = data['image'], data['label'], data['edge']

        inputs_v = inputs.type(torch.FloatTensor).to(device)
        labels_v = labels.type(torch.FloatTensor).to(device)
        edge_v = edge.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df = net(inputs_v)
        loss1, loss = muti_bce_loss_fusion(g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df, labels_v, edge_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss1.data.item()

        # del temporary outputs and loss
        del g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df, loss1, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
    Loss.append(running_tar_loss / ite_num4val)
    if (epoch+1) % 20 == 0 and (epoch+1) >= 200 :           # save model every 50 epochs
        torch.save(net.state_dict(), model_dir + "EMINet_epoch_%d_trnloss_%3f_priloss_%3f.pth" % ((epoch+1), running_loss/ite_num4val, running_tar_loss/ite_num4val))
    running_loss = 0.0
    running_tar_loss = 0.0
    net.train()  # resume train
    ite_num4val = 0

LOSS = torch.tensor(Loss)
torch.save(LOSS,'loss')
print('-------------Congratulations! Training Done!!!-------------')
