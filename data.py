import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    def __getitem__(self,index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)
def getDataLoader():
    # 读取csv文件
    df = pd.read_csv("../train.csv")
    # 将数据转换为numpy数组
    data = df.iloc[:, :5].values
    label = df.iloc[:, -1].values
    # 将数据转换为torch tensor
    data = torch.FloatTensor(data)
    label = torch.LongTensor(label)
    # 创建自定义数据集
    my_dataset = MyDataset(data, label)
    # 创建dataloader
    dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
    return dataloader
