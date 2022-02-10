import time

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import pickle
import sys
import csv
import os
import os.path as osp
import shutil
import pandas as pd
from IPython.display import display, HTML
from torchattacks import *
import datetime
import hashlib
from initializeDatasets import encryptFilesAndStore, decryptFilesAndVerify
from itertools import chain, combinations
import threading
#from threading import Thread, Lock
from multiprocessing import Process, Pool, Pipe, Lock



# Load other libraries here.
# Keep it minimal! We should be easily able to reproduce your code.
# We only support sklearn and pytorch.
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data as data
import torchvision

# We preload pytorch as an example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

undefended_model_path = "undefended_model.pth"
threshold_model_path = "threshold_model.pth"
defended_model_path = "defended_model.pth"
whiteboxRandomModel = "whiteboxRandomModel.pth"

substituteModelName0 = "substituteModel0.pth"
substituteModelName1 = "substituteModel1.pth"
substituteModelName2 = "substituteModel2.pth"

untrainedModelPath = "untrainedModel.pth"
attackDict = {} #filled by main
listAdvModels = []
n_epochs = 10
batch_size_train = 64
batch_size_test = 1
learning_rate_1 = 0.01
learning_rate_2 = 0.05
momentum_1 = 0.5
momentum_2 = 0.25
log_interval = 10

device = torch.device('cpu')

path = os.path.abspath(__file__)
wspacepath = os.path.dirname(path)
pathElements = wspacepath.split(os.sep)
wspacepath = os.sep.join(pathElements)
pathElements.extend(["_out"])
outDir = os.sep.join(pathElements)


# (2)define model
# Model with no dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # 16*24*24
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),  # 32*20*20
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32*10*10
            nn.Conv2d(32, 64, 5),  # 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*3*3
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 64 * 3 * 3)
        out = self.fc_layer(out)

        return out


class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Fully connected neural network with one hidden layer
class FeedForwardNeuralNet(nn.Module):
    def __init__(self):
        super(FeedForwardNeuralNet, self).__init__()
        self.input_size = 784
        self.l1 = nn.Linear(784, 500)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


class MultiPerceptronNet(nn.Module):
    def __init__(self):
        super(MultiPerceptronNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x

# (3)define loss, optimizer
# Training function for all networks
def train(modelInp, optimizerInp, epoch, saveModel, train_loader, attackGeneratorList=None, criterionInp=None, flgReshape = False ):
    # defining list to save training and testing loss for future evaluation.
    train_losses = []
    train_counter = []
    modelInp.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if flgReshape:
            data = data.reshape(-1, 28 * 28)

        data = data.to(device)
        target = target.to(device)

        optimizerInp.zero_grad()
        output = modelInp(data)
        if criterionInp is None:
            loss = F.nll_loss(output, target)
        else:
            loss = criterionInp(output, target)
        loss.backward()
        optimizerInp.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(modelInp.state_dict(), saveModel)
    accuracy = 100 * correct / len(train_loader.dataset)
    train_acc.append(accuracy)
    print("Epoch = {}, Training Accuracy = {}".format(epoch, accuracy))
    return accuracy


def test(modelInp, test_loader_arg, attackGeneratorList=None, attackNameList='', id=0, lock=None, flgReshape = False):
    begin_time = datetime.datetime.now()
    modelInp.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    for dataX, target in test_loader_arg:
        if flgReshape:
            dataX = dataX.reshape(-1, 28 * 28)
        dataX = dataX.to(device)
        target = target.to(device)

        dataTmp = dataX
        if attackGeneratorList is not None:
            for attackType in attackGeneratorList:
                try:
                    dataTmp = attackType(dataTmp, target)
                except:
                    #print("dataTmp shape before reshape call", dataTmp.shape)
                    dataTmp = attackType(dataTmp.reshape(-1, 28*28), target)
                    #print("dataTmp shape after attackType call", dataTmp.shape)
                    dataTmp = dataTmp.reshape(1, 1, 28, 28)
                    #print("dataTmp shape after reshape call", dataTmp.shape)

        output = modelInp(dataTmp)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    try:
        accuracy = 100 * correct / len(test_loader_arg.dataset)
    except:
        accuracy = 100 * correct / len(test_loader_arg)

    try:
        test_loss /= len(test_loader_arg.dataset)
    except:
        test_loss /= len(test_loader_arg)
    test_losses.append(test_loss)

    runTime = (datetime.datetime.now() - begin_time)
    print("Attacking with: {} \n Test accuracy: {}\n Execution time: {} \n ################################################################################"
          .format(attackNameList, accuracy, runTime))

    return accuracy


def recordTestParams(id, attackSet, accuracy, runTime, lock):
    global dictTestParams

    if lock:
        lock.acquire()
        dictTestParams[id] = (attackSet, accuracy, runTime)
        lock.release()
    else:
        print("recordTestParams has not lock over dictTestParams, therefore storing skipped")

def dumpTestParams(jsonName):
    global dictTestParams
    with open(os.sep.join([outDir, jsonName]), 'w', encoding='utf-8') as dumpFile:
        json.dump(dictTestParams, dumpFile, ensure_ascii=False, indent=4)

    dictTestParams = {}


def evaluateAttacks(originalModel, lock):

    global attackDict, attackDictKeys
    attackThreadPoolDict = {}
    attackKeys = attackDictKeys

    # Initializing threading helper dict
    for i in range(len(attackKeys)):
        attackThreadPoolDict[i + 1] = []

    for itr, attackSet in enumerate(
            chain.from_iterable(combinations(attackKeys, r) for r in range(len(attackKeys) + 1)), 1):
        if attackSet:
            print("Generated attack set: {}".format(attackSet))
            attackModelList = []
            attackNameList = []

            for subsModelNr in range(3):
                for attackX in list(attackSet):
                    attackNameList.append(str(attackX) + str(subsModelNr))
                    attackModelList.append(attackDict[subsModelNr][attackX])
            # trying combinatorial attacks
            procInst = Process(target=test,
                               args=(originalModel, test_loader, attackModelList, attackNameList, itr, lock))

            attackThreadPoolDict[len(list(attackSet))].append(procInst)

    print("Starting threads for generating adversarial samples... \n .....The command line might be dead for a while")
    # Threads are used to execute the adversarial set computation.
    # Threads are grouped with the number of cascaded attacks in order to make the join more efficient
    # A simple asscending order of attack depth would also work.
    # Attack depth = number of cascaded layers in the attack
    for attackDepth in list(attackThreadPoolDict.keys()):
        print("Generating adversarial samples for {} cascaded attacks".format(attackDepth))
        for threadX in attackThreadPoolDict[attackDepth]:
            threadX.start()
        print("################################################################################")
        for threadX in attackThreadPoolDict[attackDepth]:
            threadX.join()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dictTestParamsMutex = Lock()
    np.random.seed(200)
    imageDataForPlotting = []
    dictTestParams = {}
    train_acc = []
    test_acc = []
    fileDecryptEnabled = False

    # torchvision dataloaders to download MNIST dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    dataSetPath = './dataManualDownload/original'

    if fileDecryptEnabled:
        if decryptFilesAndVerify(dataSetPath):
            raise Exception("Datasets could not be validated")
        else:
            print("Datasets authenticated. Starting training....")

    dataset = datasets.MNIST(root=dataSetPath, train=[True, False], transform=transform, download=True)
    trainSet, testSet = torch.utils.data.random_split(dataset, [50000, 10000])

    train_loader = data.DataLoader(trainSet, batch_size=batch_size_train, shuffle=True)
    test_loader = data.DataLoader(testSet, batch_size=batch_size_test, shuffle=True)

    networkOriginal = Target().to(device)
    optimizerOriginal = optim.SGD(networkOriginal.parameters(), lr=learning_rate_2, momentum=momentum_2)

    print("################################################################################")
    print("Training original model")
    print("################################################################################")

    for epoch in range(1, n_epochs + 1):
        train_acc.append(train(networkOriginal, optimizerOriginal, epoch, undefended_model_path, train_loader))
        test_acc.append(test(networkOriginal, test_loader))


    print("Training Summary - ")
    for epoch in range(1, n_epochs + 1):
        print('.....Epoch %d, Train Accuracy: %f, Test Accuracy: %f' % (epoch, train_acc[epoch - 1], test_acc[epoch - 1]))
    print("################################################################################")

    epsilonSmall = 0.05
    epsilonMiddle = 0.1
    epsilonLarge = 0.15

    epsilonUse = epsilonMiddle

    # Substitute model 0 - Multi layer perceptron
    print("################################################################################")
    print("Training MultiPerceptronNet model")
    print("################################################################################")
    substituteModel0 = MultiPerceptronNet().to(device)
    criterionSub0 = nn.CrossEntropyLoss()
    optimizerSub0 = torch.optim.SGD(substituteModel0.parameters(),lr = 0.01)

    #train(substituteModel0, optimizerSub0, 4, substituteModelName0, train_loader, criterionInp=criterionSub0)
    train_acc = []
    test_acc = []
    for epoch in range(1, n_epochs + 1):
        train_acc.append(train(substituteModel0, optimizerSub0, epoch, substituteModelName0, train_loader, criterionInp=criterionSub0))
        test_acc.append(test(substituteModel0, test_loader))

    print("Training Summary - ")
    for epoch in range(1, n_epochs + 1):
        print(
            '.....Epoch %d, Train Accuracy: %f, Test Accuracy: %f' % (epoch, train_acc[epoch - 1], test_acc[epoch - 1]))
    print("################################################################################")





    print("################################################################################")
    print("Training FeedForwardNeuralNet model")
    print("################################################################################")
    #Substitute model 1 - Feed-Forward Neural Network
    substituteModel1 = FeedForwardNeuralNet().to(device)
    criterionSub1 = nn.CrossEntropyLoss()
    optimizerSub1 = torch.optim.Adam(substituteModel1.parameters(), lr=0.001)

    #train(substituteModel1, optimizerSub1, 4, substituteModelName1, train_loader, criterionInp=criterionSub1)
    train_acc = []
    test_acc = []
    for epoch in range(1, n_epochs + 1):
        train_acc.append(train(substituteModel1, optimizerSub1, epoch, substituteModelName1, train_loader,
                               criterionInp=criterionSub1, flgReshape=True))
        test_acc.append(test(substituteModel1, test_loader, flgReshape=True))

    print("Training Summary - ")
    for epoch in range(1, n_epochs + 1):
        print(
            '.....Epoch %d, Train Accuracy: %f, Test Accuracy: %f' % (epoch, train_acc[epoch - 1], test_acc[epoch - 1]))
    print("################################################################################")


    print("################################################################################")
    print("Training resnet50 model")
    print("################################################################################")
    #Substitute model 2 - Resnet CNN model
    substituteModel2 = models.resnet50(pretrained=True)
    substituteModel2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    substituteModel2.fc = nn.Linear(2048, 10, bias=True)

    print("Testing Summary - ")
    test(substituteModel2, test_loader)
    print("################################################################################")

    #Attack dict for Substitute model 0
    attackDictSubsModel0 = {
        "DeepFool": DeepFool(substituteModel0, steps=10),  # used to be 1000 steps
        "FGSM": FGSM(substituteModel0, eps=epsilonUse),
        "CW": CW(substituteModel0, c=100, lr=0.01, steps=10, kappa=10),  # used to be 1000 steps,
        "PGD": PGD(substituteModel0, eps=epsilonUse, alpha=0.5, steps=7, random_start=True),
    }

    # Attack dict for Substitute model 1
    attackDictSubsModel1 = {
        "DeepFool": DeepFool(substituteModel1, steps=10),  # used to be 1000 steps
        "FGSM": FGSM(substituteModel1, eps=epsilonUse),
        "CW": CW(substituteModel1, c=100, lr=0.01, steps=10, kappa=10),  # used to be 1000 steps,
        "PGD": PGD(substituteModel1, eps=epsilonUse, alpha=0.5, steps=7, random_start=True),
    }

    # Attack dict for Substitute model 2
    attackDictSubsModel2 = {
        "DeepFool": DeepFool(substituteModel2, steps=10),  # used to be 1000 steps
        "FGSM": FGSM(substituteModel2, eps=epsilonUse),
        "CW": CW(substituteModel2, c=100, lr=0.01, steps=10, kappa=10),  # used to be 1000 steps,
        "PGD": PGD(substituteModel2, eps=epsilonUse, alpha=0.5, steps=7, random_start=True),
    }

    attackDictKeys = ["DeepFool", "FGSM", "CW", "PGD"]
    attackDict = [attackDictSubsModel0, attackDictSubsModel1, attackDictSubsModel2]


    evaluateAttacks(originalModel=networkOriginal,
                    lock=dictTestParamsMutex)


    exit() # explicit exit to prevent temporary code below from executing


