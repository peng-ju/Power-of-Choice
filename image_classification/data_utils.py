import os
import json
import random
import math
from random import Random

import numpy as np
from numpy.random import RandomState

import torch
from torch.utils.data import Dataset, Subset, random_split
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms

def read_data(train_data_dir, test_data_dir):
    ''' parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('train.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('test.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


class SyntheticDataset(Dataset):
    def __init__(self, data_dir, train=True):
        if train:
            _, _, self.data, _ = read_data(data_dir, data_dir)
        else: 
            _, _, _, self.data = read_data(data_dir, data_dir)
        
        # since the dataset is already partitioned, we maintain an index_map
        # that maps (user_i, index_i) -> global_index
        self.data_indices = {}
        self.partitions = {}
        count = 0
        for uname in sorted(self.data.keys()):
            for i in range(len(self.data[uname]['x'])):
                uid = int(uname.split('_')[1])  # 'f_00001' -> 1
                self.data_indices[count] = (uid, i)
                if uid in self.partitions:
                    self.partitions[uid].append(count)
                else:
                    self.partitions[uid] = [count]
                count += 1

    def __getitem__(self, index):
        uid, i = self.data_indices[index]
        uname = 'f_{0:05d}'.format(int(uid))  # 1 -> 'f_00001'
        x = torch.tensor(self.data[uname]['x'][i], dtype=torch.float32)
        y = torch.tensor(self.data[uname]['y'][i], dtype=torch.int64)
        return x, y

    def __len__(self):
        return len(self.data_indices)
    

class FederatedDataset(object):
    def __init__(self, dataset, num_clients=100, seed=1234, rnd=0, 
                 isNonIID=False, alpha=0.2, subset_ratio=None):
        print('Loading data: %s' % dataset
              + ', num_clients: %d' % num_clients
              + ', seed: %d' % seed
              + ', rnd: %d' % rnd
              + ', isNonIID: %s' % isNonIID
              + ', alpha: %s' % alpha
              + ', subset_ratio: %s' % subset_ratio)
        self.dataset = dataset

        if dataset == 'synthetic':
            # data
            self.trainset = SyntheticDataset('../data/synthetic_data/', train=True)
            self.testset = SyntheticDataset('../data/synthetic_data/', train=False)

            # partitions
            self.train_partitions = self.trainset.partitions
            self.test_partitions = self.testset.partitions

            # ratio
            self.ratio = np.array([len(v) for k, v in self.train_partitions.items()])
            self.ratio = self.ratio/np.sum(self.ratio)
            print('TRAIN Data ratio: %s' % str(self.ratio))
            print('sum of ratio: %s' % str(sum(self.ratio)))

            # input size
            x, y = self.trainset[0]
            self.input_dim = x.size(0)

        elif dataset == 'synthetic-all':
            ## full synthetic data (no partitions)
            ## simulates a single client with all data: baseline for traditional ML
            # data
            self.trainset = SyntheticDataset('../data/synthetic_data/', train=True)
            self.testset = SyntheticDataset('../data/synthetic_data/', train=False)

            # partitions
            self.train_partitions = {0: [j for i in self.trainset.partitions.values() for j in i]}  # self.trainset.partitions
            self.test_partitions = {0: [j for i in self.testset.partitions.values() for j in i]}  # self.testset.partitions

            # ratio
            self.ratio = np.array([len(v) for k, v in self.train_partitions.items()])
            self.ratio = self.ratio/np.sum(self.ratio)
            print('TRAIN Data ratio: %s' % str(self.ratio))
            print('sum of ratio: %s' % str(sum(self.ratio)))

            # input size
            x, y = self.trainset[0]
            self.input_dim = x.size(0)

        else:
            if dataset == 'fmnist':
                # data
                apply_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
                self.trainset = torchvision.datasets.FashionMNIST(root='../data',
                                                        train=True,
                                                        download=True,
                                                        transform=apply_transform)

                self.testset = torchvision.datasets.FashionMNIST(root='../data',
                                                    train=False,
                                                    download=True,
                                                    transform=apply_transform)

            elif dataset == 'cifar':
                # TODO: add data partitions
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                self.trainset = torchvision.datasets.CIFAR10(root='../data',
                                                    train=True, 
                                                    download=True, 
                                                    transform=transform_train)
                
                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                self.testset = torchvision.datasets.CIFAR10(root='../data',
                                                train=False, 
                                                download=True, 
                                                transform=transform_test)

            elif dataset == 'emnist':
                # TODO: add data partitions
                apply_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
                self.trainset = torchvision.datasets.EMNIST(root='../data',
                                                        split = 'digits',
                                                        train=True,
                                                        download=True,
                                                        transform=apply_transform)

                self.testset = torchvision.datasets.EMNIST(root='../data',
                                                            split= 'digits',
                                                            train=False,
                                                            download=True,
                                                            transform=apply_transform)
                
            # subset data 
            # TODO: add this functionality for synthetic data ??
            if subset_ratio:
                self.trainset, _ = random_split(self.trainset, [subset_ratio, 1-subset_ratio])
                self.testset, _ = random_split(self.testset, [subset_ratio, 1-subset_ratio])
                
            # partitions
            partition_sizes = [1.0 / num_clients for _ in range(num_clients)]
            partitioner = DataPartitioner(self.trainset, partition_sizes, seed, rnd, 
                                                isNonIID, alpha, dataset)
            self.train_partitions = partitioner.partitions
            partitioner_ = DataPartitioner(self.testset, partitioner.ratio, seed, rnd, 
                                                False, 0, dataset)
            self.test_partitions = partitioner_.partitions
            
            # ratio
            self.ratio = partitioner.ratio  # Ratio of data sizes
            print('Data ratio: %s' % str(self.ratio))
            
            # input dim
            self.input_dim = np.prod(self.trainset[0][0].shape)
            print('input_dim:', self.input_dim)

        return


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chunks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], rnd=0, seed=1234, isNonIID=False, alpha=0,
                 dataset=None, print_f=50):
        self.data = data
        self.dataset = dataset

        if isNonIID:
            self.partitions, self.ratio, self.dat_stat, self.endat_size = self.__getDirichletData__(data, sizes,
                                                                                                    alpha, rnd, print_f)

        else:
            self.partitions = [] 
            self.ratio = sizes
            rng = Random() 
            rng.seed(seed) # seed is fixed so same random number is generated
            data_len = len(data) 
            indexes = [x for x in range(0, data_len)] 
            rng.shuffle(indexes)    # Same shuffling (with each seed)

            for frac in sizes: 
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed, alpha):
        # labelList = data.targets
        labelList = np.array([data[i][1] for i in range(len(data))])  # alternative for above
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]

        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum

        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))

        # majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        majorLabelNumPerPartition = 2
        basicLabelRatio = alpha
        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]

        return partitions

    def __getDirichletData__(self, data, psizes, alpha, rnd, print_f):
        n_nets = len(psizes)
        K = 10
        # labelList = np.array(data.targets)
        labelList = np.array([data[i][1] for i in range(len(data))])  # alternative for above
        min_size = 0
        N = len(labelList)
        rann = RandomState(2020)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                rann.shuffle(idx_k)
                proportions = rann.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            rann.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)

        if rnd % print_f == 0:
            print('Data statistics: %s' % str(net_cls_counts))
            print('Data ratio: %s' % str(weights))

        return idx_batch, weights, net_cls_counts, np.sum(local_sizes)
