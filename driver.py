
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_ import log_string, plot_train_val_loss
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test

import torch

config = {'time_slot': 5,
        'num_his': 5,
        'num_pred': 3,
        'L': 1,
        'K': 8,
        'd': 8,
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'batch_size': 4,
        'max_epoch': 1,
        'patience': 10,
        'learning_rate': 0.001,
        'decay_epoch': 10,
        'traffic_file': './data/pems-bay.h5',
        'SE_file': './data/SE(PeMS).txt',
        'model_file': './data/GMAN.pkl',
        'log_file': './data/log'}

class Config():
    def __init__(self, dict_):
        self.__dict__ = dict_

args = Config(config)

#%%
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE, mean, std
# build model
log_string(log, 'compiling model...')

model = GMAN(SE, args, bn_decay=0.1)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))


#%%


if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    plot_train_val_loss(loss_train, loss_val, 'figure/train_val_loss.png')
    trainPred, valPred, testPred = test(args, log)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
    trainPred_ = trainPred.numpy().reshape(-1, trainY.shape[-1])
    trainY_ = trainY.numpy().reshape(-1, trainY.shape[-1])
    valPred_ = valPred.numpy().reshape(-1, valY.shape[-1])
    valY_ = valY.numpy().reshape(-1, valY.shape[-1])
    testPred_ = testPred.numpy().reshape(-1, testY.shape[-1])
    testY_ = testY.numpy().reshape(-1, testY.shape[-1])

    # Save training, validation and testing datas to disk
    l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    for i, data in enumerate(l):
        np.savetxt('./figure/' + name[i] + '.txt', data, fmt='%s')
        
    # Plot the test prediction vs target（optional)
    plt.figure(figsize=(10, 280))
    for k in range(325):
        plt.subplot(325, 1, k + 1)
        for j in range(len(testPred)):
            c, d = [], []
            for i in range(12):
                c.append(testPred[j, i, k])
                d.append(testY[j, i, k])
            plt.plot(range(1 + j, 12 + 1 + j), c, c='b')
            plt.plot(range(1 + j, 12 + 1 + j), d, c='r')
    plt.title('Test prediction vs Target')
    plt.savefig('./figure/test_results.png')


