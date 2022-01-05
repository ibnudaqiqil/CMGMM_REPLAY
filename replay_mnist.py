# some initial imports
import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from models.MFA import MFA
from models.MNIST import FNet
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_model, samples_to_np_images

from continualai.colab.scripts import mnist
mnist.init()


use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(1)

# print some MNIST info
x_train, t_train, x_test, t_test = mnist.load()

print(f'Size of training data is: {t_train.shape[0]}, size of test data: {t_test.shape[0]}')


# #############################################################################
def train(model, device, x_train, t_train, optimizer, epochs=10, log_training=False):
    for epoch in range(epochs):
        model.train()

        for start in range(0, len(t_train)-1, 256):
            end = start + 256
            x, y = torch.from_numpy(x_train[start:end]).float(
            ), torch.from_numpy(t_train[start:end]).long()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            if log_training:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def test(model, device, x_test, t_test):
    model.eval()
    test_loss = 0
    correct = 0
    for start in range(0, len(t_test)-1, 256):
        end = start + 256
        with torch.no_grad():
            x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(
                t_test[start:end]).long()
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(t_test)
    return test_loss, 100. * correct / len(t_test)

def shuffle_in_unison(dataset, seed, in_place=False):
    np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


def train_singlehead(past_examples_percentage=0, epochs=15):
  model = FNet().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  accs_fine_grid = []
  for i in range(tasks_num):
    x_train, t_train, _, _ = task_data[i]

    # concatenate previous tasks
    for j in range(0, i + 1):
        past_x_train, past_t_train, _, _ = task_data[j]
        example_num = int(past_examples_percentage * len(past_t_train))
        x_train = np.concatenate((x_train, past_x_train[:example_num]))
        t_train = np.concatenate((t_train, past_t_train[:example_num]))

    x_train, t_train = shuffle_in_unison([x_train, t_train], 0)
    print(x_train.shape)
    train(model, device, x_train, t_train, optimizer, epochs)

    # test on tasks seen so far
    accs_subset = []
    for j in range(0, i + 1):
        _, _, x_test, t_test = task_data[j]
        _, test_acc = test(model, device, x_test, t_test)
        print("acc=", i, "==>", j, "==>", test_acc)
        accs_subset.append(test_acc)

    if i < (tasks_num - 1):
        accs_subset.extend([np.nan] * (4 - i))

    accs_fine_grid.append(accs_subset)

  return accs_fine_grid


def train_singleHead_Gmm(past_examples_percentage=0, gmm_concept=[]):
    model = FNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    accs_fine_grid = []
    for i in range(tasks_num):
        x_train, t_train, _, _ = task_data[i]

        # concatenate previous tasks
        for j in range(0, i + 1):
            print("generate task =", i, "==>", j)
            past_x_train, past_t_train, _, _ = task_data[j]
            example_num = int(past_examples_percentage * len(past_t_train))
            #print("generating concept => ",task_classes_arr[i][0], "Total : ", example_num)
            digits_a, _ = gmm_concept[task_classes_arr[i][0]].sample(example_num)
            digits_b, _ = gmm_concept[task_classes_arr[i][1]].sample(example_num)
            #digits_b = pca_concept[task_classes_arr[i][1]].inverse_transform(b[0])*10
            #print(digits_a.shape, digits_b.shape)
            np.clip(digits_a, 0, 255, out=digits_a)
            #print(digits_a.shape, digits_b.shape)
            digits_a[np.abs(digits_a) < 0.001] = 0
            digits_a[np.abs(digits_a) > 0.001] = digits_a[np.abs(digits_a) > 0.001] + 0.3
            aa = digits_a.astype(np.float32)
            aa = aa.reshape(aa.shape[0], 1, 28, 28)

            np.clip(digits_b, 0, 255, out=digits_b)
            digits_b[np.abs(digits_b) < 0.001] = 0
            digits_b[np.abs(digits_b) > 0.001] = digits_b[np.abs(digits_b) > 0.001] + 0.3

            bb = digits_b.astype(np.float32)

            bb = bb.reshape(aa.shape[0], 1, 28, 28)

            #plt.imshow(aa[0][0])

            a_x = np.full((example_num), task_classes_arr[i][0])
            b_x = np.full((example_num), task_classes_arr[i][1])
            

            x_train = np.concatenate((x_train, aa, bb))
            t_train = np.concatenate((t_train, a_x, b_x))

        x_train, t_train = shuffle_in_unison([x_train, t_train], 0)
        print(x_train.shape)
        train(model, device, x_train, t_train, optimizer, 15)

        # test on tasks seen so far
        accs_subset = []
        for j in range(0, i + 1):
            _, _, x_test, t_test = task_data[j]
            _, test_acc = test(model, device, x_test, t_test)
            print("acc=", i, "==>", j, "==>", test_acc)
            accs_subset.append(test_acc)

        if i < (tasks_num - 1):
            accs_subset.extend([np.nan] * (4 - i))

        accs_fine_grid.append(accs_subset)

    return accs_fine_grid


def train_gmm(train_set,  model_name, epochs=15):
    #model_dir ="models"
    #print(train_set.shape)
    from os import path

    if path.exists(model_name+".pickle"):
        print("Loading model from file:", model_name+".pickle")
        f = open(model_name+".pickle", "rb")
        model = pickle.load(f)
        f.close()
        return model
        #load pickle file
    else:  # Download Dataset
       
        train_set = train_set.reshape((train_set.shape[0], -1))

        model = GaussianMixture(n_components = 50, init_params='kmeans',
                n_init = 5, max_iter = 5000, covariance_type = 'diag')
        model.fit(train_set)
        f = open(model_name+'.pickle', 'wb')

        pickle.dump(model, f)


        print ('Model saved in path: PATH_TO'+model_name)

    return model


concept_num = [0,1,2,3,4,5,6,7,8,9]
# tasks to use
task_classes_arr = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
tasks_num = len(task_classes_arr)  # 5

task_data = []
gmm_concept=[]
task_data_with_overlap = []
for i, task_classes in enumerate(task_classes_arr):
    train_mask = np.isin(t_train, task_classes)
    test_mask = np.isin(t_test, task_classes)
    x_train_task, t_train_task = x_train[train_mask], t_train[train_mask]
    x_test_task, t_test_task = x_test[test_mask], t_test[test_mask]

    task_data.append((x_train_task, t_train_task, x_test_task, t_test_task))
    task_data_with_overlap.append((x_train_task, t_train_task - (i * 2),
                                   x_test_task, t_test_task - (i * 2)))
#train or reload GMM
for i, task_classes in enumerate(concept_num):
    train_mask = np.isin(t_train, task_classes)
    test_mask = np.isin(t_test, task_classes)

    x_train_task, t_train_task = x_train[train_mask], t_train[train_mask]
    x_test_task, t_test_task = x_test[test_mask], t_test[test_mask]
    #print(x_train_task[0][0])
    gmm = train_gmm(x_train_task,  "model_"+str(task_classes))
    gmm_concept.append(gmm)

#accs_naive = train_singlehead()
accs_rehearsal_all = train_singlehead(0.8)


#accs_fine_grid = np.array(accs_naive)
#nan_mask = np.isnan(accs_naive)
#sns.heatmap(accs_rehearsal_all, vmin=0, vmax=100, mask=nan_mask, annot=True,  fmt='g',
#            yticklabels=range(1, 6), xticklabels=range(1, 6), ax=axes[2], cbar=False)
