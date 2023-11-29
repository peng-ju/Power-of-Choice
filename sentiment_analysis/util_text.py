import numpy as np
from random import Random
import random
import logging
import torch
import torch.utils.data as data_utils

from models import *
# from params import args_parser

from data_preprocessing import *

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logging.debug('This message should appear on the console')

# args = args_parser()


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

    def __init__(self, data, dataset=None):
        self.data = data
        self.dataset = dataset

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(size, args, rnd):

    if args.dataset == "twitter":
        print("begin load Glove twitter embedding \n")
        # load GloVe embeddings
        word2vectors, word2id = load_GloVe_twitter_emb()
        print("finish load Glove twitter embedding \n")
        # load the twitter dataset and splits in train/val/test
        train, test, partition, ratio = load_twitter_datasets(args.minimum_tweets, args) # train and test
        Xtrain, Ytrain = processAllTweets2vec(train, word2vectors)
        Xtest, Ytest = processAllTweets2vec(test, word2vectors)

        Xtrain, Ytrain = torch.from_numpy(Xtrain), torch.from_numpy(Ytrain)
        Xtest, Ytest = torch.from_numpy(Xtest), torch.from_numpy(Ytest)

        train_data = data_utils.TensorDataset(Xtrain.type(torch.FloatTensor), Ytrain.type(torch.LongTensor))
        test_data = data_utils.TensorDataset(Xtest.type(torch.FloatTensor), Ytest.type(torch.LongTensor))

        train_loader = data_utils.DataLoader(train_data, shuffle=False, batch_size=64, num_workers=0)
        test_loader = data_utils.DataLoader(test_data, shuffle=False, batch_size=64, num_workers=0)


    return partition, train_loader, test_loader, ratio, train_data


def partitiondata_loader(partition, rank, batch_size, train_data):
    '''
    single mini-batch loader
    '''
    partition = partition[rank]

    data_idx = random.sample(range(len(partition)), k=int(min(batch_size, len(partition))))
    partitioned = torch.utils.data.Subset(train_data, indices=data_idx)

    trainbatch_loader = torch.utils.data.DataLoader(partitioned,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True)

    return trainbatch_loader


def sel_client(DataRatios, cli_loss, cli_val, args, rnd):
    '''
    Client selection part 
    returning the indices the set $\mathcal{S}$ and $\mathcal{A}$
    
    :param DataRatios: $p_k$
    :param cli_loss: actual local loss F_k(w)
    :param cli_val: proxy of the local loss
    :param args: variable arguments
    :param rnd: communication round index
    :return: idxs_users (indices of $\mathcal{S}$), rnd_idx (indices of $\mathcal{A}$)
    '''
    
    # If reproducibility is needed
    # rng1 = Random()
    # rng1.seed(seed)
    np.random.seed(args.seed)

    rnd_idx = []
    if args.seltype == 'rand':
        # random selection in proportion to $p_k$ with replacement
        idxs_users = np.random.choice(args.ensize, p=DataRatios, size=args.size, replace=True)

    elif args.seltype == 'randint':
        # 'rand' for intermittent client availability
        delete = 0.2 # randomly delete
        if (rnd % 2) == 0: # first half
            del_idx = np.random.choice(int(args.ensize / 2), size=int(delete * args.ensize / 2), replace=False)
            search_idx = np.delete(np.arange(0, args.ensize / 2), del_idx)
        else: # second half
            del_idx = np.random.choice(np.arange(args.ensize / 2, args.ensize), size=int(delete * args.ensize / 2),
                                       replace=False)
            search_idx = np.delete(np.arange(args.ensize / 2, args.ensize), del_idx)

        # normalize DataRatios after delete 20% clients
        idxs_users = np.random.choice(search_idx, 
                                      p=[DataRatios[int(i)] for i in search_idx] / sum([DataRatios[int(i)]
                                                                                                    for i in
                                                                                                    search_idx]),
                                      size=args.size, replace=True)

    elif args.seltype == 'pow-d':
        # standard power-of-choice strategy
        # get the candidate clients
        rnd_idx = np.random.choice(args.ensize, p=DataRatios, size=args.powd, replace=False)
        # select top k clients with largest loss
        repval = list(zip([cli_loss[i] for i in rnd_idx], rnd_idx))
        repval.sort(key=lambda x: x[0], reverse=True)
        rep = list(zip(*repval)) # unpacking operator * to unzip the data
        idxs_users = rep[1][:int(args.size)]

    elif args.seltype == 'rpow-d':
        # computation/communication efficient variant of 'pow-d'
        rnd_idx1 = np.random.choice(args.ensize, p=DataRatios, size=args.powd, replace=False)
        repval = list(zip([cli_val[i] for i in rnd_idx1], rnd_idx1))
        repval.sort(key=lambda x: x[0], reverse=True)
        rep = list(zip(*repval))
        idxs_users = rep[1][:int(args.size)]

    elif args.seltype == 'pow-dint':
        # 'pow-d' for intermittent client availability
        delete = 0.2
        if (rnd % 2) == 0:
            del_idx = np.random.choice(int(args.ensize / 2), size=int(delete * args.ensize / 2), replace=False)
            search_idx = list(np.delete(np.arange(0, args.ensize / 2), del_idx))
        else:
            del_idx = np.random.choice(np.arange(args.ensize / 2, args.ensize), size=int(delete * args.ensize / 2),
                                       replace=False)
            search_idx = list(np.delete(np.arange(args.ensize / 2, args.ensize), del_idx))

        rnd_idx = np.random.choice(search_idx, p=[DataRatios[int(i)] for i in search_idx] / sum([DataRatios[int(i)]
                                                                                                 for i in search_idx]),
                                   size=args.powd, replace=False)

        repval = list(zip([cli_loss[int(i)] for i in rnd_idx], rnd_idx))
        repval.sort(key=lambda x: x[0], reverse=True)
        rep = list(zip(*repval))
        idxs_users = rep[1][:int(args.size)]

    elif args.seltype == 'rpow-dint':
        # 'rpow-d' for intermittent client availability
        delete = 0.2
        if (rnd % 2) == 0:
            del_idx = np.random.choice(int(args.ensize / 2), size=int(delete * args.ensize / 2), replace=False)
            search_idx = list(np.delete(np.arange(0, args.ensize / 2), del_idx))
        else:
            del_idx = np.random.choice(np.arange(args.ensize / 2, args.ensize), size=int(delete * args.ensize / 2),
                                       replace=False)
            search_idx = list(np.delete(np.arange(args.ensize / 2, args.ensize), del_idx))

        rnd_idx = np.random.choice(search_idx, p=[DataRatios[int(i)] for i in search_idx] / sum([DataRatios[int(i)]
                                                                                                 for i in search_idx]),
                                   size=args.powd, replace=False)

        repval = list(zip([cli_val[int(i)] for i in rnd_idx], rnd_idx))
        repval.sort(key=lambda x: x[0], reverse=True)
        rep = list(zip(*repval))
        idxs_users = rep[1][:int(args.size)]

    elif args.seltype == 'afl':
        # benchmark strategy
        soft_temp = 0.01
        sorted_loss_idx = np.argsort(cli_val)

        for j in sorted_loss_idx[:int(args.delete_ratio * args.ensize)]:
            cli_val[j] = -np.inf

        # select part with softmax loss
        loss_prob = np.exp(soft_temp * cli_val) / sum(np.exp(soft_temp * cli_val))
        idx1 = np.random.choice(int(args.ensize), 
                                p=loss_prob, 
                                size=int(np.floor((1 - args.rnd_ratio) * args.size)),
                                replace=False)
        
        # select part randomly
        new_idx = np.delete(np.arange(0, args.ensize), idx1)
        idx2 = np.random.choice(new_idx, 
                                size=int(args.size - np.floor((1 - args.rnd_ratio) * args.size)),
                                replace=False)

        idxs_users = list(idx1) + list(idx2)


    return idxs_users, rnd_idx


def choices(population, weights=None, cum_weights=None, k=1):
    """
    Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """

    if cum_weights is None:
        if weights is None: # random without weight
            total = len(population)
            result = []
            for i in range(k):
                random.seed(i)
                result.extend(population[int(random.random() * total)]) # random index
            return result
        # calculate culmulative weigths
        cum_weights = []
        c = 0
        for x in weights:
            c += x
            cum_weights.append(c)
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    
    # validate inputs
    if len(cum_weights) != len(population):
        raise ValueError('The number of weights does not match the population')
    
    total = cum_weights[-1]
    hi = len(cum_weights) - 1
    from bisect import bisect
    result = []
    for i in range(k):
        random.seed(i)
        # cum_weights convert weights to linear space
        # bisect locate the selected position
        result.extend(population[bisect(cum_weights, random.random() * total, 0, hi)]) 
    return result


class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, 
                 init_dict=None, 
                 ptag='Time', 
                 stateful=False,
                 csv_format=True):
        
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        """initialize"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        """update the stats"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))
