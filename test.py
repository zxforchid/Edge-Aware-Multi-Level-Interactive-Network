import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader_test import RescaleT
from data_loader_test import CenterCrop
from data_loader_test import ToTensor
from data_loader_test import ToTensorLab
from data_loader_test import SalObjDataset

from model import EMINet
import timeit


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name, pred, d_dir):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]       
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')

# --------- 1. get image path and name ---------

image_dir = "/home/fh/data/NoisyImages/sp0.2/"
prediction_dir = './Data/results/'     
model_dir = "/home/fh/Edge3/trained_models/EMINet_epoch_350_trnloss_1.679714_priloss_0.005589.pth"

img_name_list = glob.glob(image_dir + '*.bmp')

# --------- 2. dataloader ---------
#1. dataload
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
#test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([transforms.Resize(256),transforms.ToTensor()]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False)

# --------- 3. model define ---------
print("...load EMINet...")
net = EMINet(3,1)
net.load_state_dict(torch.load(model_dir, map_location={'cuda:0': 'cuda:3'}))
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()

# --------- 4. inference for each image ---------
start = timeit.default_timer()
for i_test, data_test in enumerate(test_salobj_dataloader):

	print("inferencing:",img_name_list[i_test].split("/")[-1])

	inputs_test = data_test['image']
	inputs_test = inputs_test.type(torch.FloatTensor).to(device)

	# if torch.cuda.is_available():
	# 	inputs_test = Variable(inputs_test.cuda())
	# else:
	# 	inputs_test = Variable(inputs_test)

	g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df= net(inputs_test)

	# normalization
	pred = df[:,0,:,:]
	pred = normPRED(pred)

	# save results to test_results folder
	save_output(img_name_list[i_test],pred,prediction_dir)

	del g1, g2, g3, g4, g5, g6, g7, e1, e2, e3, e4, e5, e6, df
end = timeit.default_timer()
print(str(end-start))
