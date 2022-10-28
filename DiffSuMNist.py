#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
import random

from utils import download_dataset, show_batch, show

batch_size = 128
operators_embeddings = {'+': torch.rand([batch_size, 1, 28, 28]),
                        '-': torch.rand([batch_size, 1, 28, 28])}


class Model(nn.Module):
    """
    Model with 6 conv layers (3 conv layers for each input image and operator) followed by two fully connected layers after
     concatenating the results of the input images
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l1 = nn.Linear(192, 1024)
        self.l2 = nn.Linear(1024, 19)

    def forward(self, x, y, operator):
        x = self.conv_layers1(x)
        y = self.conv_layers1(y)
        operator = self.conv_layers1(operator)
        x = self.conv_layers2(x)
        y = self.conv_layers2(y)
        operator = self.conv_layers2(operator)
        x = self.conv_layers3(x)
        y = self.conv_layers3(y)
        operator = self.conv_layers3(operator)
        N, _, _, _ = x.size()
        x = x.view(N, -1)
        y = y.view(N, -1)
        operator = operator.view(N, -1)
        z = torch.cat((x, y, operator), 1)
        z = self.l1(z)
        z = self.relu(z)
        z = self.l2(z)

        return z

    def conv_layers1(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        return x

    def conv_layers2(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)
        return x

    def conv_layers3(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp(x)
        return x


def train_model(model, trainloader, testloader, trainset, testset, criterion, optimizer, use_gpu=False, num_epochs=20):
    '''
    Function to train a model while creating dataset on the go: in each iteration of len(batches),
    take two consecutive batches and treat them as input1 and input2. Also, take operator, either - and + in
    alternate manner.
    Send data (inputs1, inputs2, operator) though network and calculate loss and accuracy. Do backpropagation.
    Repeat for num_epoch.
    Return trained model.
    :param model: <class '__main__.Model'>
    :param: trainloader: batches of train images
    :param: testloader: batches of test images
    :param: trainset:  MNIST dataset
    :param: testset: MNIST dataset
    :param criterion: <class 'torch.nn.modules.loss.*>
    :param optimizer: <class 'torch.optim.*'>
    :param num_epochs: int
    :return: <class '__main__.Model'>
    '''

    since = time.time()

    err_train = []
    err_test = []

    train_acc = []
    test_acc = []

    best_acc = 0.0
    output_pred = []
    output_real = []

    for epoch in range(num_epochs):

        print('-' * 30)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 30)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            i = 0

            loader = trainloader
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                loader = testloader

            # calculate epoch loss
            running_loss = 0.0
            running_corrects = 0

            # save predicted and true labeling
            output_pred_val = []
            output_real_val = []

            # Iterate over batches
            first_batch = next(iter(loader))
            one_true = True
            for i, batch in enumerate(loader):

                next_batch = first_batch
                first_batch = batch

                if i % 2 == 0:
                    continue

                # get the inputs ,labels of two consecutive batches
                inputs1, labels1 = first_batch
                inputs2, labels2 = next_batch

                # alternate operation minus and plus
                if one_true:
                    k = 0
                    one_true = False
                else:
                    one_true = True
                    k = 1

                # wrap them in Variable
                if use_gpu:
                    inputs1 = Variable(inputs1.float().cuda())
                    labels1 = Variable(labels1.long().cuda())

                    inputs2 = Variable(inputs2.float().cuda())
                    labels2 = Variable(labels2.long().cuda())

                    if k == 0:
                        operator = Variable(operators_embeddings['+'].float().cuda())
                    else:

                        operator = Variable(operators_embeddings['-'].float().cuda())

                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels2)
                    inputs2, labels2 = Variable(inputs2), Variable(labels2)

                    if k == 0:
                        operator = Variable(operators_embeddings['+'])
                    else:
                        operator = Variable(operators_embeddings['-'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs1, inputs2, operator)
                _, preds = torch.max(outputs.data, 1)

                # if k=0 which represents subtraction, add 9
                if k == 0:
                    labels = labels1 + labels2
                else:
                    labels = labels1 - labels2 + 9

                loss = criterion(outputs, labels)

                # backprop in training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                else:
                    output_real_val = np.concatenate((output_real_val, labels.data.cpu().numpy()), 0)
                    output_pred_val = np.concatenate((output_pred_val, preds.cpu().numpy()), 0)

                # statistics

                running_loss += loss.detach()
                running_corrects += torch.sum(preds == labels.data)

            # evaluate the model according to type
            if phase == 'train':
                len_dataset = len(trainset)
            else:
                len_dataset = len(testset)

            epoch_loss = running_loss / (len_dataset / 2)
            epoch_acc = running_corrects / (len_dataset / 2)

            if phase == 'train':
                err_train.append(epoch_loss.item())
                train_acc.append(epoch_acc.item())
            else:
                err_test.append(epoch_loss.item())
                test_acc.append(epoch_acc.item())

            if phase == 'val':
                if best_acc < epoch_acc:
                    best_acc = epoch_acc
                    output_pred = output_pred_val
                    output_real = output_real_val

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    return model, err_train, err_test, train_acc, test_acc


def evaluate(model, testloader, use_gpu=False, show_wrong_results_only=False):
    '''
    Function - evaluates model on 10 randomly picked combination of images and randomly picked operator
             - shows models output
    :param model: trained model, class '__main__.Model'
    :param testloader: data loader
    :param use_gpu: bool
    :param show_wrong_results_only, bool, show only wrong results for error analysis
    :return: None
    '''

    first_batch = next(iter(testloader))

    print('len testloader', len(testloader))
    for i, batch in enumerate(testloader):

        next_batch = first_batch
        first_batch = batch

        if i % 2 == 0:
            continue

        # get the inputs ,labels of the batch
        inputs1, labels1 = first_batch
        inputs2, labels2 = next_batch

        # get random operation, either - or +
        k = random.randint(0, 1)

        # Wrap inputs and values into Variable
        if use_gpu:
            inputs1 = Variable(inputs1.float().cuda())
            labels1 = Variable(labels1.long().cuda())
            inputs2 = Variable(inputs2.float().cuda())
            labels2 = Variable(labels2.long().cuda())

            if k == 0:
                operator = Variable(operators_embeddings['+'].float().cuda())
            else:
                operator = Variable(operators_embeddings['-'].float().cuda())
        else:
            inputs1, labels1 = Variable(inputs1), Variable(labels2)
            inputs2, labels2 = Variable(inputs2), Variable(labels2)
            if k == 0:
                operator = Variable(operators_embeddings['+'])
            else:
                operator = Variable(operators_embeddings['-'])

        # forward
        outputs = model(inputs1, inputs2, operator)
        _, preds = torch.max(outputs.data, 1)

        random_example = random.randint(0, 127)
        if k == 0:
            prediction = preds[random_example].item()
            real = labels1[random_example].item() + labels2[random_example].item()
        else:
            prediction = preds[random_example].item() - 9
            real = labels1[random_example].item() - labels2[random_example].item()

        if show_wrong_results_only:
            if prediction != real:
                show(inputs1[random_example][0], inputs2[random_example][0], labels1[random_example].item(),
             labels2[random_example].item(), prediction, k)
        else:
             show(inputs1[random_example][0], inputs2[random_example][0], labels1[random_example].item(),
             labels2[random_example].item(), prediction, k)


if __name__ == '__main__':

    # trainset, testset = download_dataset()
    # numbers from 0 to 27 represent classes of numbers from -9 to 18
    classes = [str(i) for i in range(0, 28)]
    # show_batch(trainset)

    # Download mnist training dataset and create samples/batches
    trainset = torchvision.datasets.MNIST('./datasets/MNIST_data', train=True, download=True,
                                          transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=batch_size)

    # download mnist validation dataset and sample it
    testset = torchvision.datasets.MNIST('./datasets/MNIST_data', train=False, download=True,
                                         transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=batch_size)

    # check if GPU is available
    use_gpu = torch.cuda.is_available()
    print('GPU available: ', use_gpu)

    model = Model()
    if use_gpu:
        model = model.cuda()

    # define criterion and optimizer function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    trained_model, err_train, err_test, train_acc, test_acc = train_model(model, trainloader, testloader,
                                                                          trainset, testset,
                                                                          criterion, optimizer,
                                                                          use_gpu=use_gpu)


    evaluate(trained_model, testloader, use_gpu=use_gpu)
