import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    def __getitem__(self,index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

def prepare_data():
    # 读取csv文件
    train_df = pd.read_csv("data/train.csv",header=None)
    test_df = pd.read_csv("data/test.csv",header=None)

    # 将数据转换为numpy数组
    train_data_all = train_df.iloc[:, :5].values
    train_target_all = train_df.iloc[:, -1].values
    test_data = test_df.iloc[:, :5].values
    test_target = test_df.iloc[:, -1].values

    # 将数据转换为torch tensor
    train_data_all = torch.FloatTensor(train_data_all)
    train_target_all = torch.LongTensor(train_target_all)
    test_data = torch.FloatTensor(test_data)
    test_target = torch.LongTensor(test_target)


    # 归一化
    data_mean = torch.mean(train_data_all, dim=0)
    data_std = torch.std(train_data_all, dim=0)
    train_data_all = (train_data_all - data_mean) / data_std
    test_data = (test_data - data_mean) / data_std

    shuffler_idx = torch.randperm(train_target_all.size(0))
    train_data_all = train_data_all[shuffler_idx]
    train_target_all = train_target_all[shuffler_idx]

    train_data = []
    train_target = []

    train_data_pool = train_data_all[15000:60000, :]
    train_target_pool = train_target_all[15000:60000]

    train_data_pool = train_data_pool.float()

    train_data_all = train_data_all.float()
    test_data = test_data.float()

    #每种类别随机从数据集中选出100张作为训练集
    for i in range(0, 2):
        arr = np.array(np.where(train_target_all.numpy() == i))
        idx = np.random.permutation(arr)
        # 随机选出label为i的10张图片 为 data_i
        data_i = train_data_all.numpy()[idx[0][0:3000], :]
        target_i = train_target_all.numpy()[idx[0][0:3000]]
        train_data.append(data_i)
        train_target.append(target_i)
    # 上面这段循环的作用：从数据集中每种类别随机选10张 组成train_data
    train_data = np.concatenate(train_data, axis=0).astype("float32")
    train_target = np.concatenate(train_target, axis=0)

    # 最终，train_data_pool为从数据集中选了450000 而train_data只有6000张

    # 创建自定义数据集
    my_dataset = MyDataset(train_data, train_target)
    test_dataset = MyDataset(test_data,test_target)
    # 创建dataloader
    train_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)

    return torch.from_numpy(train_data).float(), torch.from_numpy(train_target), \
           train_data_pool, train_target_pool, \
           test_data, test_target, \
           train_dataloader,test_loader

