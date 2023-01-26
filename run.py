from VAEACGAN_Train import VAEACGAN
from data import prepare_data
from activeLearn import acquire_points
from config import Config
if __name__ == '__main__':
    best_acc = 0
    config = Config()
    args = config.args
    train_data, train_target, pool_data, pool_target, \
    test_data, test_target,train_loader,test_loader = prepare_data(args)
    gan = VAEACGAN(args,train_loader,test_loader) #包含了vae生成模型和分类模型
    best_acc = gan.train(best_acc) #首先训练一下vae里面的普通分类模型
    #主动学习获取数据集 然后利用对抗训练
    pool_data,pool_target = acquire_points(args,gan,train_data,train_target,pool_data,pool_target,best_acc)
    print ("Training over,best acc is ",best_acc)