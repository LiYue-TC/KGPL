import pdb

from sklearn.model_selection import train_test_split

# pytorch package
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
from tqdm import tqdm
from utility.myutility import random_sample, tensor_to_img, save_img
from layers.GraphConvolution import GraphConvolution
from layers.MultiHeadGAT import MultiHeadGAT
from layers.AttentionCrop import AttentionCropLayer
import cv2
import os
import numpy as np
import pandas as pd
import logging
import datetime
from model import KPGL

# from tiny_vit import TinyViT
batch_size = 256
epochs = 200
patient = 200
learning_rate = 0.001
test_batch = 512
in_channel = 1
chunk_interval = 1
model_name = 'model_saver_final'
# train_save_name = 'TFG-Net_training_stage'
root_path = '/home/ubuntu/HDDs/HDD1/ly/TI_Estimation-master'
train_range = (2010, 2017)
test_range = (2018, 2020)
#%%
model_dir = './'+model_name + '/'
data_info = pd.read_csv(root_path + '/data/TC_processed_data_regression_12.csv')
min_max_img = np.load(root_path + '/data/gridsat.img.min.max.npy')
#%%
min_img = min_max_img[0]
max_img = min_max_img[1]


test_info = data_info[(data_info["YEAR"]>=test_range[0]) & (data_info["YEAR"]<=test_range[1])].copy()

X = test_info["IMAGE_PATH"].to_numpy()
label = test_info["LABEL"].to_numpy().reshape(-1,1)

Text = test_info["WIND_SPEED"].to_numpy().reshape(-1,1)
Y = np.concatenate((label,Text),axis = 1)


test_list = []
pred_list = []

def crop_center(matrix, crop_width):
    total_width = matrix.shape[1]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return matrix[:, start:end, start:end]


def divide_and_resize(image):

    _, height, width = image.shape
    target_size = (height, width)
    image = image.transpose(1, 2, 0)
    sub_size = height // 6
    #  定义裁剪位置
    center = image[sub_size * 2 : sub_size * 4, sub_size * 2:sub_size * 4, :]  # 中心裁剪
    top = image[0 : sub_size * 2, sub_size * 2 : sub_size * 4, :]  # 顶部裁剪
    bottom = image[sub_size * 4 : sub_size * 6, sub_size * 2 : sub_size * 4, :] # 底部裁剪
    left = image[sub_size * 2 : sub_size * 4, 0 : sub_size * 2, :]  # 左侧裁剪
    right = image[sub_size * 2 : sub_size * 4, sub_size * 4 : sub_size * 6]  # 右侧裁剪
    # images = [top]
    images = [center, top, bottom, left, right]
    # 循环处理每个图像
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], target_size).transpose(2, 0, 1)

    images.append(image.transpose(2, 0, 1))
    crop_images = np.concatenate([images], axis=0)
    # pdb.set_trace()
    return crop_images

class LiverDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        # self.transform = transform
        self.key = transform
        if  self.key:
            self.transform = transforms.Compose([
            # transforms.ToTensor(),  # 转换为张量
            transforms.RandomRotation(20)  # 随机旋转，角度范围为-20到+20度
        ])



    def __getitem__(self, index):
        img = np.load(X[self.data[index]])
        img_normal = (img - min_img) / (max_img - min_img)
        # pdb.set_trace()
        img = crop_center(img_normal,224)
        # pdb.set_trace()
        img = np.concatenate([img] * 3, axis=0)
        crop_image = divide_and_resize(img)
        return crop_image.astype(np.float32), Y[self.data[index]].astype(np.float32)

    def __len__(self):
        return len(self.data)



test_indexset = LiverDataset([*range(0, len(X))],transform = False)
test_index_loader = DataLoader(dataset=test_indexset,batch_size=64,shuffle=False,)


model = TFGNet().cuda()


checkpoint = torch.load(model_dir + '/' + 'model.pth')
model.load_state_dict(checkpoint['model'])
model = model.to("cuda:0")
model.eval()
with torch.no_grad():
    for _data,_target in tqdm(test_index_loader):
        data = _data.cuda()
        # target= _target.cuda()
        text = _target[:, 1].reshape(-1, 1).cuda()
        target = _target[:, 0].reshape(-1, 1).cuda()
        pred,other = model(data,text)
        target = target.cpu().detach().numpy()
        pred_list.extend(pred.cpu().detach().numpy())
        test_list.extend(target)


index_2018 = 0
index_2019 = 0
index_2020 = 0
for i in range(len(X)):
    if int(X[i].split('/')[-2][:4])==2018:
        index_2018 = i
    elif int(X[i].split('/')[-2][:4])==2019:
        index_2019 = i
    elif int(X[i].split('/')[-2][:4])==2020:
        index_2020 = i
index_2018 += 1
index_2019 += 1
index_2020 += 1


def get_mae(test, pred):
    result = []
    for i in range(len(test)):
        result.append(np.abs(test[i]-pred[i])[0])
    return sum(result)/len(result)
#%%
def get_rmse(test, pred):
    result = []
    for i in range(len(test)):
        result.append(np.square(test[i] - pred[i])[0])

    mse = sum(result) / len(result)
    rmse = np.sqrt(mse)
    return rmse


rmse_2018 = get_rmse(test_list[:index_2018], pred_list[:index_2018])
rmse_2019 = get_rmse(test_list[index_2018:index_2019], pred_list[index_2018:index_2019])
rmse_2020 = get_rmse(test_list[index_2019:index_2020], pred_list[index_2019:index_2020])
rmse = get_rmse(test_list, pred_list)
print('rmse in 2018:', rmse_2018)
print('rmse in 2019:', rmse_2019)
print('rmse in 2020:', rmse_2020)
print('rmse in avg:', rmse)

