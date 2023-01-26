from torch.autograd import Variable
from model import *
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from eval import test
import torch

#二元分类 focal loss
class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

def latent_loss(mu, log_var):

    std = log_var.mul(0.5).exp_()
    mean_sq = mu * mu
    stddev_sq = std * std
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    # # Alternative
    # return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

def reparameterize(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

class VAEACGAN(object):
    def __init__(self,args,train_loader,test_loader):
        # parameters
        self.num_classes = args.num_classes #分类类别数
        self.args = args
        self.pre_epochs = args.pre_epochs
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.gpu_mode = args.gpu_mode
        self.model_name = 'VAEACGAN'

        self.focal_loss = FocalLoss()

        self.model = Lenet() #分类器
        self.E = Encoder()          # Encoder
        self.G = Generator()        # Generator/Decoder
        self.D = Discriminator()    # Discriminator

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.c_lr)

        self.E_optimizer = optim.Adadelta(self.E.parameters(), lr=args.e_lr, rho=0.9, eps=1e-6)
        self.G_optimizer = optim.Adadelta(self.G.parameters(), lr=args.g_lr, rho=0.9, eps=1e-6)
        self.D_optimizer = optim.Adadelta(self.D.parameters(), lr=args.d_lr, rho=0.9, eps=1e-6)

        self.E_scheduler = lr_scheduler.StepLR(self.E_optimizer, step_size=8, gamma=0.5, last_epoch=-1)
        self.G_scheduler = lr_scheduler.StepLR(self.G_optimizer, step_size=8, gamma=0.5, last_epoch=-1)
        self.D_scheduler = lr_scheduler.StepLR(self.D_optimizer, step_size=8, gamma=0.5, last_epoch=-1)

        if self.gpu_mode:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.G.to(self.device)
            self.D.to(self.device)
            self.E.to(self.device)
            self.BCE_loss = nn.BCELoss().to(self.device)
            self.CE_loss = nn.CrossEntropyLoss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

        self.train_loader = train_loader
        self.test_loader = test_loader
    def train(self,best_acc): #普通分类网络训练
        self.model.train()
        loss = None
        criterion = nn.CrossEntropyLoss()
        tot_loss = 0
        for epoch in range(self.pre_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.gpu_mode:
                    data, target = data.to(self.device), target.to(self.device)
                data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()
                output = self.model(data)
                #focal loss
                loss = self.focal_loss(output,target)
                #loss = criterion(output, target)
                tot_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            tot_loss /= len(self.train_loader)

            if epoch % 10 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, tot_loss))
                test_loss,best_acc = test(self.model,epoch,self.test_loader,best_acc)
            tot_loss = 0
        return best_acc
    def antagonisticTrain(self,best_acc):#对抗训练

        self.train_hist = {}
        self.train_hist['E_loss'] = []
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['C_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.z_dim = 100
        self.y_dim = 10

        #[100,1] real全1 fake全0
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).to(self.device)), Variable(torch.zeros(self.batch_size, 1).to(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        for epoch in range(self.epochs):
            self.G.train()
            self.E_scheduler.step()
            self.G_scheduler.step()
            self.D_scheduler.step()
            t_C_loss = 0
            t_E_loss = 0
            t_D_loss = 0
            t_G_loss = 0
            for iter,(x_,labels) in enumerate(self.train_loader):

                #对label进行ont-hot编码 注意这里二分类
                y_vec_ = np.zeros((len(labels), self.num_classes), dtype=np.float)
                for i, label in enumerate(labels):
                    y_vec_[i, label] = 1

                y_vec_ = torch.from_numpy(y_vec_).type(torch.FloatTensor)

                z_ = torch.randn((x_.size(0), self.z_dim))

                if self.gpu_mode:
                    x_, y_vec_, z_ = Variable(x_.to(self.device)), Variable(y_vec_.to(self.device)), Variable(z_.to(self.device))
                else:
                    x_, y_vec_, z_ = Variable(x_), Variable(y_vec_), Variable(z_)

                # Fix G, update E network
                self.E_optimizer.zero_grad()

                mu, log_var = self.E(x_) #将输入送进去编码 得到 [100,1,100] [100,1,100] 就是产生一个分布
                noise = reparameterize(mu, log_var) #根据这个分布得到一个码
                noise = noise.view(x_.size(0), 100) #[100,100]
                output = self.G(noise, y_vec_) #将这个码送到生成器里面产生一个batch

                # Compute the decoder loss that will be added to network E
                ll = latent_loss(mu, log_var)
                E_loss = self.MSE_loss(output, x_)  # / self.batch_size
                E_loss += ll

                self.train_hist['E_loss'].append(E_loss.item())

                E_loss.backward(retain_graph=True)
                self.E_optimizer.step() #更新编码器部分

                # Fix E, D, C, update G network
                self.G_optimizer.zero_grad()

                # Compute the GAN loss that will be added to the Generator G
                G_ = self.G(z_, y_vec_)

                D_fake = self.D(G_)
                C_fake = self.model(G_)

                G_loss= self.BCE_loss(D_fake, self.y_real_[0:x_.size(0),:])
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss

                # Compute the decoder loss that will be added to the Generator G

                mu, log_var = self.E(x_)
                noise = reparameterize(mu, log_var)
                noise = noise.view(x_.size(0), 100)
                G_dec = self.G(noise, y_vec_)

                G_dec_loss = self.MSE_loss(G_dec, x_)  # / self.batch_size

                G_loss += 0.75 * G_dec_loss

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # Fix G, update D, C network

                self.D_optimizer.zero_grad()
                self.optimizer.zero_grad()

                D_real = self.D(x_)
                C_real = self.model(x_)

                D_real_loss = self.BCE_loss(D_real, self.y_real_[0:x_.size(0),:])
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)

                D_fake = self.D(G_)
                C_fake = self.model(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_[0:x_.size(0),:])
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                mu, log_var = self.E(x_)
                noise = reparameterize(mu, log_var)
                noise = noise.view(x_.size(0), 100)
                # output = self.G(noise, y_vec_)

                G_dec = self.G(noise, y_vec_)

                D_dec = self.D(G_dec)
                C_dec = self.model(G_dec)
                D_dec_loss = self.BCE_loss(D_dec, self.y_fake_[0:x_.size(0),:])
                C_dec_loss = self.CE_loss(C_dec, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + D_fake_loss + D_dec_loss
                C_loss = C_real_loss + C_fake_loss + C_dec_loss

                self.train_hist['D_loss'].append(D_loss.item())
                self.train_hist['C_loss'].append(C_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                C_loss.backward(retain_graph=True)
                self.optimizer.step()

                t_C_loss += C_loss.item()
                t_E_loss += E_loss.item()
                t_D_loss += D_loss.item()
                t_G_loss += G_loss.item()
            print("Epoch: [%2d] D_loss: %.8f, G_loss: %.8f, C_loss: %.8f, E_loss: %.8f" %
                  ((epoch + 1), t_D_loss / len(self.train_loader), t_G_loss / len(self.train_loader)
                   , t_C_loss / len(self.train_loader), t_E_loss / len(self.train_loader)))
            self.model.eval()
            _, best_acc = test(self.model,self.epochs,self.test_loader,best_acc)
        return best_acc


