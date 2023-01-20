import random
import torch
import numpy as np
from data import MyDataset
from torch.utils.data import Dataset, DataLoader
import time
from eval import evaluate
from scipy.stats import mode
#主动学习实现 model为待训练的分类模型
def acquire_points(args,vae,
                   train_data,train_target,
                   pool_data,pool_target,
                   best_acc,
                   random_sample=False):

    model = vae.model #分类模型
    acquisition_iterations = 100
    dropout_iterations = 20  # [50, 100, 500, 1000]
    Queries = 100
    nb_samples = 100
    pool_all = np.zeros(shape=(1))

    acquisition_function = variation_ratios_acquisition #TODO 目前仅实现VAR_RATIOS查询函数

    test_acc_hist = []
    for i in range(acquisition_iterations):
        pool_subset = 2000
        if random_sample:
            pool_subset = Queries
        print('---------------------------------')
        print ("Acquisition Iteration " + str(i))
        # 随机从数据池子中选出 2000 张图片
        pool_subset_dropout = torch.from_numpy(np.asarray(random.sample(range(0, pool_data.size(0)), pool_subset),dtype=np.longlong))
        pool_data_dropout = pool_data[pool_subset_dropout]
        pool_target_dropout = pool_target[pool_subset_dropout]
        if random_sample is True:
            pool_index = np.array(range(0, Queries))

        else:
            points_of_interest = acquisition_function(args,model,dropout_iterations, pool_data_dropout, pool_target_dropout)
            pool_index = points_of_interest.argsort()[-Queries:][::-1]

        pool_index = torch.from_numpy(np.flip(pool_index, axis=0).copy())

        pool_all = np.append(pool_all, pool_index)

        pooled_data = pool_data_dropout[pool_index]  # LongTensor
        pooled_target = pool_target_dropout[pool_index]  # LongTensor

        train_data = torch.cat((train_data, pooled_data), 0)
        train_target = torch.cat((train_target, pooled_target), 0)
        #remove from pool set
        pool_data,pool_target = remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index,pool_data,pool_target)

        # Train the ACGAN here
        print("times:{},Train the ACGAN".format(i + 1))
        test_acc = vae.antagonisticTrain(best_acc);
        test_acc_hist.append(test_acc)
        print("test_acc_hist:",test_acc)
        #  TODO 可视化结果，gan.visualize_results(epochs)
    return pool_data,pool_target
    #TODO 保存模型 np.save("./test_acc_VAEACGAN_MNIST" + argument + ".npy", np.asarray(test_acc_hist))

def variation_ratios_acquisition(args,model,dropout_iterations, pool_data_dropout, pool_target_dropout):
    # print("VARIATIONAL RATIOS ACQUSITION FUNCTION")
    All_Dropout_Classes = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = MyDataset(pool_data_dropout,pool_target_dropout)
    # 创建dataloader
    pool_loader = DataLoader(pool, batch_size=args.batch_size, shuffle=True)

    start_time = time.time()
    #一次dropout_iter 就评估一次pool_loader中的数据集，总共评估dropout_iterations次
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(model,pool_loader, stochastic=True, predict_classes=True)

        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
    # print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Variation = np.zeros(shape=(pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        Predicted_Class, Mode = mode(L[1:]) #计算本次iter中 出现次数最多的classes以及出现次数
        v = np.array([1 - Mode / float(dropout_iterations)])
        Variation[t] = v #这个样本在所有iter中的权重值
    points_of_interest = Variation.flatten()
    return points_of_interest

def remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index,pool_data,pool_target):
    np_data = pool_data.numpy()
    np_target = pool_target.numpy()
    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()
    np_index = pool_index.numpy()
    np.delete(np_data, pool_subset, axis=0)
    np.delete(np_target, pool_subset, axis=0)

    np.delete(pool_data_dropout, np_index, axis=0)
    np.delete(pool_target_dropout, np_index, axis=0)

    np_data = np.concatenate((np_data, pool_data_dropout), axis=0)
    np_target = np.concatenate((np_target, pool_target_dropout), axis=0)

    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)

    return pool_data,pool_target



