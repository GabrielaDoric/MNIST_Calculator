from __future__ import print_function, division
import torchvision
from PIL import Image
import random
import numpy as np
from matplotlib import pyplot as plt






def show_batch(trainset):
    len_dataset = len(trainset)

    classes = [str(i) for i in range(0, 10)]
    random_list = random.sample(range(len_dataset), 9)
    plt.rcParams['axes.grid'] = False
    # loop over the indices and plot the images as sub-figures
    j = 0
    for i in random_list:
        img, label = trainset[i]
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = np.array(img)
        plt.subplot(3, 3, j + 1)
        plt.title(classes[label])
        plt.subplots_adjust(top=1.5, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)
        plt.imshow(img, cmap='gray')

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        j += 1
    plt.show()


def download_dataset():
    trainset = torchvision.datasets.MNIST('./datasets/MNIST_data', train=True, download=True)
    # print('Length of training set ', len(trainset))
    testset = torchvision.datasets.MNIST('./datasets/MNIST_data', train=False, download=True)
    # print('Length of testseting set ', len(testset))

    return trainset, testset

def show(img1, img2, label1, label2, label, k):
    fig = plt.figure()
    plt.axis('off')
    plt.rcParams['axes.grid'] = False
    plt.grid(b=None)
    if k == 0:
        operation = 'Addition'
        operator = '+'
    if k == 1:
        operation = 'Substraction'
        operator = '-'

    output_str = str(operation) + ':' + str(label1) + str(operator) + str(label2) + '=' + str(label)
    print(output_str)

    plt.title(output_str)
    a = fig.add_subplot(1, 2, 1)

    npimg1 = img1.cpu().data.numpy()
    npimg2 = img2.cpu().data.numpy()

    plt.imshow(npimg1, cmap='gray')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(npimg2, cmap='gray')

    plt.show()
