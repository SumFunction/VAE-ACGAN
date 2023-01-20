import argparse
class Config():
    def __init__(self):
        parser = argparse.ArgumentParser()
        '''
        训练数据的相关参数
        '''
        parser.add_argument('--epochs', type=int,default=100, help="训练的迭代次数")
        parser.add_argument('--num_classes', type=int, default=2, help="分类类别数")
        parser.add_argument('--batch_size', type=int, default=64, help="训练的批次大小")
        parser.add_argument('--num_workers', type=int, default=0, help="读取数据的num_wokers数量")
        parser.add_argument('--gpu_mode',type=bool,default=True,help="是否启用GPU")
        parser.add_argument('--lr', default=2.5e-4, type=int, help="learing rate in training")
        self.args = parser.parse_args()
    def print_and_return(self):
        print("---------输出训练的参数-------------")
        print("---batch_size:",self.args.batch_size)
        print("---iters:",self.args.iters)
        print("---val_step",self.args.val_step)
        print("----------------------------------")
