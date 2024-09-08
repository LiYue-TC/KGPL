# package to split train and validate set
import pdb
import cv2
from sklearn.model_selection import train_test_split
import random
# pytorch package
import torch
import timm
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
from tqdm import tqdm
from utility.myutility import random_sample, tensor_to_img, save_img
from layers.GraphConvolution import GraphConvolution
from layers.MultiHeadGAT import MultiHeadGAT
from layers.AttentionCrop import AttentionCropLayer
# import math
# from sklearn.metrics import mean_squared_error
import os
import ast
import numpy as np
import pandas as pd
# import logging
import datetime
# from model import TFGNet
# from hcvarr import HCVARR
# from tiny_vit import TinyViT
from model_all_test import TFGNet
from torchvision import transforms
from PIL import Image
# from conformer import Conformer
batch_size = 64
epochs = 50
patient = 200
learning_rate = 0.00005
test_batch = 512
in_channel = 1
chunk_interval = 1
model_name = 'model_saver_final'
train_save_name = 'TFG-Net_training_stage'
root_path = '/home/ubuntu/HDDs/HDD1/ly/TI_Estimation-master'
train_range = (2010, 2017)
test_range = (2018, 2020)
#%%
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
model_dir = './'+model_name + '/'
data_info = pd.read_csv(root_path + '/data/TC_processed_data_regression.csv')
#%%
train_info = data_info[(data_info["YEAR"]>=train_range[0]) & (data_info["YEAR"]<=train_range[1])].copy()
#%%
X = train_info["IMAGE_PATH"].to_numpy()
# pdb.set_trace()
label = train_info["LABEL"].to_numpy().reshape(-1,1)
# pdb.set_trace()
# label_list_str = train_info["LABEL_SET"][1]
# label_list = ast.literal_eval(label_list_str)  # 解析字符串表示的列表

# 将列表转换为 NumPy 数组
# Y = np.array(label_list).reshape(-1, 1)
Text = train_info["WIND_SPEED"].to_numpy().reshape(-1,1)

Y = np.concatenate((label,Text),axis = 1)
# Y = np.expand_dims(Y, axis=1)
# text_list_str = train_info["LABEL_SET"][0]
# text_list = ast.literal_eval(label_list_str)
# Text = np.array(text_list ).reshape(-1, 1)#
# Text = np.expand_dims(Text, axis=1)
#%%
min_max_img = np.load(root_path + '/data/gridsat.img.min.max.npy')
#%%
min_img = min_max_img[0]
max_img = min_max_img[1]


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

train_valid_index = [*range(0, len(X))]
len(train_valid_index)
# %%
# split training data and validation data
train_index, valid_index = train_test_split(train_valid_index, test_size=0.2, shuffle=True)

# check data shape
print("training index shape:", len(train_index))
print("validation index shape:", len(valid_index))

# form dataset
train_indexset = LiverDataset(train_index,transform=False)
valid_indexset = LiverDataset(valid_index,transform=False)

# form dataloader
train_index_loader = DataLoader(dataset=train_indexset, batch_size=batch_size, shuffle=True, )
valid_index_loader = DataLoader(dataset=valid_indexset, batch_size=batch_size, shuffle=True, )

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
    images = [center, top, bottom, left, right]
    # images = [top]
    # 循环处理每个图像
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], target_size).transpose(2, 0, 1)

    images.append(image.transpose(2, 0, 1))
    crop_images = np.concatenate([images], axis=0)
    # pdb.set_trace()
    return crop_images

def RankingLoss(pred):
    # pdb.set_trace()
    label = torch.full_like(pred[0], fill_value= 1)
    criterion = torch.nn.MarginRankingLoss(margin=0.0)
    loss = []
    for i in range(1, len(pred)):
        loss.append(criterion(pred[0], pred[i], label))
    # # loss_tensor = torch.cat(loss)
    # # pdb.set_trace()
    loss_sum = loss[0] + loss[1] + loss[2]
    # loss_sum = criterion(pred[1], pred[0], label)
    return loss_sum

#  train

model = TFGNet()
# model.load_state_dict(torch.load('./model_all/model_best.pth'))
# model = TinyViT()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
start_epoch = 0
min_val_loss = float('inf')
# model = nn.DataParallel(model,device_ids=[0, 1, ])
# model.to("cuda:0")
model.to("cuda:0")
start = datetime.datetime.now()
sample = random_sample(valid_index_loader)

for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
    print('epochs [%d/%d]' % (epoch + 1, epochs))

    print('epochs [%d]' % (optimizer.param_groups[0]['lr']))
    # logging.info('epochs [' + str(epoch + 1) + '/' + str(epochs) + ']')
    starttime = datetime.datetime.now()
    # train data
    total_train_loss = 0
    train_cnt = 0
    for _data, _target in tqdm(train_index_loader):
        data = _data.cuda()
        text = _target[:,1].reshape(-1,1).cuda()
        target =_target[:,0].reshape(-1,1).cuda()
        # pdb.set_trace()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        # pred, _, _, _ = model.forward(data)
        pred,region_outs = model(data,text)
        optimizer.zero_grad()
        loss1 = criterion(pred, target)
        # pdb.set_trace()
        # label = torch.full_like(center, fill_value=-1)
        loss2 = RankingLoss(region_outs)
        loss = loss1 + loss2

        total_train_loss += loss.item()
        train_cnt += 1
        loss.backward()
        optimizer.step()
    # scheduler.step()
        # scheduler.step()
    # current_learning_rate = optimizer.param_groups[0]['lr']
    # print(f"Current Learning Rate: {current_learning_rate}")

    total_val_loss = 0
    val_cnt = 0
    for _data,_target in tqdm(valid_index_loader):
        data = _data.cuda()
        text = _target[:,1].reshape(-1,1).cuda()
        target = _target[:,0].reshape(-1,1).cuda()
        # zero the parameter gradients
        # forward + backward + optimize
        # pred, _, _, _ = model.forward(data)
        pred,region_outs = model(data,text)
        optimizer.zero_grad()
        loss1 = criterion(pred, target)
        loss2 = RankingLoss(region_outs)
        loss = loss1 + loss2
        val_cnt += 1
        total_val_loss += loss.item()

    #
    # if epoch >= 40:
    #     checkpoint = {
    #         'epoch': epoch + 1,  # next epoch
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'min_val_loss': total_val_loss
    #     }
    #     model_path = f"model_all_epoch_{epoch + 1}.pth"

        # torch.save(model.state_dict(), model_path)


    if min_val_loss > total_val_loss:
        checkpoint = {
                    'epoch': epoch + 1,  # next epoch
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'min_val_loss': total_val_loss
                }
        model_path = f"wide_4_deep_7_2.pth"
        torch.save(checkpoint, model_dir + '/' + model_path)
        min_val_loss = total_val_loss
        print('Saving Model in epoch', epoch + 1)
        # logging.info('Saving Model in epoch ' + str(epoch + 1))
        # min_val_loss = total_val_loss
    # early_stop_counter = 0

    endtime = datetime.datetime.now()
    # _, _, _, resized = model(sample.unsqueeze(0))
    # x1 = resized[0].data * (max_img - min_img) + min_img
    #
    # save_img(x1, path=f'{root_path}{train_save_name}/epoch_{epoch}@2x.jpg',
    #          annotation=f'epoch = {epoch: .1f} total_val_loss = {total_train_loss / train_cnt:.2f}, val_loss = {total_val_loss / val_cnt:.2f}')

    print('cost:%fs total_train_loss: %.8f val_loss: %.8f' % (
    (endtime - starttime).seconds, total_train_loss / train_cnt, total_val_loss / val_cnt))

end = datetime.datetime.now()
print('Finished Training')
print('Total cost: %fs' % (end - start).seconds)

