import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#分类网络
class Lenet(nn.Module):
    def __init__(self, input_dim: int=5, num_classes: int=2):
        super(Lenet, self).__init__()
        self.fc0 = nn.Linear(input_dim, 128)
        self.fc1 = nn.Linear(128 , 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(128,num_classes)
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

#编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.output_dim = 100

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=5,out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1, 1024),
            nn.ReLU()
        )
        self.fc21 = nn.Sequential(
            nn.Linear(1024,self.output_dim),
            nn.Sigmoid(),
        )
        self.fc22 = nn.Sequential(
            nn.Linear(1024,self.output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        #[batch,5]
        x = self.fc0(x)
        x = self.fc1(x)
        x = x.view(-1,1,1024)
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var

#生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder_dim = 100
        self.input_dim = self.encoder_dim + 2 # 2是类别 对标minist 10
        self.output_dim = 5

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        #由于是一维的点，类似反卷积的还原操作使用fc来代替还原通道即可
        self.deconv = nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = self.deconv(x)
        return x

#判别器
class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_dim = 5
        self.output_dim = 1

        self.conv = nn.Sequential( #类比原论文鉴别器的conv
            nn.Linear(self.input_dim,64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 , 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.fc1(x)
        d = self.dc(x)
        return d
