import argparse
class Config():
    def __init__(self):
        parser = argparse.ArgumentParser()
        '''
        训练数据的相关参数
        '''
        parser.add_argument('--pre_epochs', type=int, default=100, help="预训练迭代次数")
        parser.add_argument('--pre_size', type=int, default=5000, help="预训练的数据集大小")

        parser.add_argument('--epochs', type=int,default=100, help="训练的迭代次数")
        parser.add_argument('--num_classes', type=int, default=2, help="分类类别数")
        parser.add_argument('--batch_size', type=int, default=64, help="训练的批次大小")
        parser.add_argument('--num_workers', type=int, default=0, help="读取数据的num_wokers数量")
        parser.add_argument('--gpu_mode',type=bool,default=True,help="是否启用GPU")
        parser.add_argument('--c_lr', default=0.001, type=int, help="分类器学习率")
        parser.add_argument('--e_lr', default=0.01, type=int, help="编码器学习率")
        parser.add_argument('--g_lr', default=0.01, type=int, help="生成器学习率")
        parser.add_argument('--d_lr', default=0.01, type=int, help="鉴别器学习率")

        parser.add_argument('--l1',default=1,type=float,help="主动学习选择的参数1")
        parser.add_argument('--l2', default=0.01, type=float, help="主动学习选择的参数2")


        parser.add_argument('--random_sample', default=False, type=bool, help="是否开启主动学习") #默认开启，如果设置为true则每次随机从pool中选取Queries大小进行训练
        parser.add_argument('--pool_subset', default=2000, type=int, help="主动学习选择评估的pool大小") #必须比下面这个参数大
        parser.add_argument('--Queries', default=100, type=int, help="主动学习最终选择的数据集大小")
        parser.add_argument('--acquisition_iterations', default=100, type=int, help="主动学习训练次数")
        parser.add_argument('--dropout_iterations', default=20, type=int, help="贝叶斯评估次数")
        self.args = parser.parse_args()
    def print_and_return(self):
        print("---------输出训练的参数-------------")
        print("---batch_size:",self.args.batch_size)
        print("---iters:",self.args.iters)
        print("---val_step",self.args.val_step)
        print("----------------------------------")
