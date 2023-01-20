#评估分类模型
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def evaluate(model,input_data, stochastic=False, predict_classes=False):

    if stochastic:
        model.train()  # we use dropout at test time
    else:
        model.eval()

    predictions = []
    test_loss = 0
    correct = 0
    cuda = True
    for data, target in input_data:
        if cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)

        softmaxed = F.softmax(output.cpu())

        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(), axis=-1))
        else:
            predictions.extend(softmaxed.data.numpy())
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        pred = pred.eq(target.data).cpu().data.float()
        correct += pred.sum()
    return test_loss, correct, predictions

def test(model,epoch,test_loader,best_acc):

    test_loss = 0
    correct = 0

    test_loss, correct, _ = evaluate(model,test_loader, stochastic=False)

    test_loss /= len(test_loader)  # loss function already averages over batch size
    test_acc = 100. * correct / len(test_loader.dataset)

    if epoch:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    if test_acc > best_acc:
        print('Saving...') #TODO 后期可保存模型
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        best_acc = test_acc

    return test_loss, best_acc