import logging
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import models
from data_utils import FederatedDataset

class FedAvg(object):
    def __init__(self, lr, bs, localE, algo, model, powd, num_clients, clients_per_round, 
                 dataset, num_classes, NIID, alpha, subset_ratio, delete_ratio, rnd_ratio, seed, device=None):
        """ initialize federated optimizer """
        # hyperparameters
        self.lr = lr  # learning rate
        self.bs = bs  # batch size
        self.localE = localE  # local epochs
        self.algo = algo  # client selection algorithm
        self.powd = powd  # d (power of choice param)
        self.clients_per_round = clients_per_round  # clients per round, m
        self.num_clients = num_clients  # len(self.train_data.keys())  # number of clients, K
        self.dataset = dataset
        self.num_classes = num_classes  # number of classes in the dataset
        self.NIID = NIID
        self.alpha = alpha
        self.subset_ratio = subset_ratio
        self.delete_ratio = delete_ratio
        self.rnd_ratio = rnd_ratio
        self.seed = seed
        self.device = device  # TODO: yet to integrate ____.to(device)

        # read data
        self.data = FederatedDataset(self.dataset, self.num_clients, self.seed, 0, 
                                     self.NIID, self.alpha, self.subset_ratio)
        self.ratio = self.data.ratio  # ratio, p_k for each client k
        self.dim = self.data.input_dim  # input dimension
        
        # define model, criterion and initialize global parameters
        if model == 'LR':
            self.model = torch.nn.Linear(self.dim, self.num_classes, bias=False).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss()
            
            # we initialize to zeros for reproducibility of original paper
            # though similar behavior if initialized to random values
            self.global_parameters = []
            with torch.no_grad():
                for param in self.model.parameters():
                    self.global_parameters.append(torch.zeros_like(param))
        
        elif model == 'MLP' or model == 'CNN':
            if model == 'MLP':
                self.model = models.MLP_FMNIST(dim_in=self.dim, dim_hidden1=64, dim_hidden2=30, dim_out=self.num_classes).to(self.device)
            elif model == 'CNN':
                self.model = models.CNN_Cifar(self.num_classes).to(self.device)

            # defining loss function
            self.criterion = torch.nn.NLLLoss()

            # neural network doesn't train if initialized to zeros
            # thus, we initialize to random values
            self.global_parameters = []
            with torch.no_grad():
                for param in self.model.parameters():
                    self.global_parameters.append(param.detach().clone())

        # defining the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=10e-4)


    def set_params(self, parameters):
        """ set parameters of global model """
        with torch.no_grad():
            for model_param, param in zip(self.model.parameters(), parameters):
                model_param.copy_(param)
    

    def get_params(self):
        """ get parameters of global model """
        parameters = []
        with torch.no_grad():
            for model_param in self.model.parameters():
                parameters.append(model_param.detach().clone())
        return parameters


    def eval(self, model, dataloader, criterion, device):
        """ compute loss, acc for client `i` on train data """
        model.eval()

        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                
                # foward pass
                y_hat = model(X)

                # compute loss
                loss_tmp = criterion(y_hat, y)
                loss += loss_tmp.item() * X.size(0)

                # prediction
                _, pred_labels = torch.max(y_hat,1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum((pred_labels == y).float()).item()

                total += X.size(0)
        loss /= total
        acc = correct/total
        return loss, acc

    def evaluate(self):
        """ evaluate global and local metrics on train data """
        global_loss = 0
        global_acc = 0
        local_losses = []
        client_comptime = []
        local_acc = []

        # compute loss for each client
        for i in range(self.num_clients):
            comptime_start = time.time()

            # fetch data for client `i`
            dataset = self.data.trainset
            partitions = self.data.train_partitions
            # NOTE: cpow-d should be active only for train data (not test data)
            if self.algo == 'cpow-d' or self.algo == 'cpow-dint':
                indices = random.sample(range(len(partitions[i])), k=min(self.bs, len(partitions[i])))
            else:
                indices = partitions[i]
            datasubset = Subset(dataset, indices=indices)
            dataloader = DataLoader(datasubset,
                                    batch_size=len(datasubset),
                                    shuffle=True,
                                    pin_memory=True)

            loss_i, acc_i = self.eval(self.model, dataloader, self.criterion, self.device)
            
            client_comptime.append(time.time() - comptime_start)
            global_loss += loss_i * self.ratio[i]
            global_acc += acc_i * self.ratio[i]
            local_losses.append(loss_i)
            local_acc.append(acc_i)

        return global_loss, global_acc, local_losses, local_acc, client_comptime
    
    def evaluate_approx(self):
        """ compute global metrics for test data """
        self.model.eval()

        # fetch full data
        dataset = self.data.testset
        dataloader = DataLoader(dataset,
                                batch_size=min(self.bs, len(dataset)),
                                shuffle=True,
                                pin_memory=True)

        loss, acc = self.eval(self.model, dataloader, self.criterion, self.device)

        return loss, acc

    def train(self, i, update=True):
        """ compute loss, acc for client `i` on train data and run optimizer step """
        self.model.train()

        # fetch single mini-batch for client `i` (stochasticity)
        datasubset = Subset(self.data.trainset, indices=self.data.train_partitions[i])
        dataloader = DataLoader(datasubset,
                                batch_size=min(self.bs, len(datasubset)),
                                shuffle=True,
                                pin_memory=True)
        X, y = next(iter(dataloader))
        X, y = X.to(self.device), y.to(self.device)

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass
        y_hat = self.model(X)

        # compute loss
        loss = self.criterion(y_hat, y)

        # prediction
        _, pred_labels = torch.max(y_hat,1)
        pred_labels = pred_labels.view(-1)
        acc = torch.mean((pred_labels == y).float())

        # backward pass - compute gradients
        loss.backward()

        # gradient norm
        grad_norm = 0
        for param in self.model.parameters():
            grad_norm += torch.norm(param.grad.detach())**2
        grad_norm = grad_norm**(1/2)

        if update:
            # backward pass - update weights
            self.optimizer.step()

        return loss.item(), acc.item(), grad_norm

    def local_update(self, active_clients):
        """ train the set of active clients """
        client_params = []
        comm_update_times = []
        losses = []
        for i in active_clients:
            comm_update_start = time.time()
            loss_i = 0

            # send global parameters to client `i`
            self.set_params(self.global_parameters)
            
            # run E steps of SGD on client `i`
            for _ in range(self.localE):
                tmp_loss, tmp_acc, tmp_norm = self.train(i)
                loss_i += tmp_loss
            loss_i = loss_i/self.localE
            losses.append(loss_i)
            
            # get local parameters from client `i`
            local_parameters = self.get_params()
            client_params.append(local_parameters)

            # track communication time
            comm_update_times.append(time.time() - comm_update_start)

        return client_params, losses, comm_update_times

    def aggregate(self, client_params):
        """ aggregation strategy in FedAvg """
        with torch.no_grad():
            # zero out the global parameters for accumulation
            for param in self.global_parameters:
                param.zero_()
            
            # sum local parameters
            for local_params in client_params:
                for global_param, param in zip(self.global_parameters, local_params):
                    global_param.add_(param)
                    # option 2: similar behaviour but slightly different
                    # global_param.add_(param, alpha=1/len(client_params))
            
            # divide local parameters
            for global_param in self.global_parameters:
                global_param.div_(len(client_params))


    def select_clients(self, client_loss, client_loss_proxy, rnd):
        '''
        Client selection part returning the indices the set $\mathcal{S}$ and $\mathcal{A}$
        Assumes that we have the list of local loss values for ALL clients

        :param data_ratios: $p_k$
        :param cli_loss: actual local loss F_k(w)
        :param cli_val: proxy of the local loss
        :param rnd: communication round index
        :return: idxs_users (indices of $\mathcal{S}$), rnd_idx (indices of $\mathcal{A}$)
        '''
        rnd_idx = []
        if client_loss == []:
            # For the first round, select 'm' clients uniformly at random
            idxs_users = np.random.choice(self.num_clients, size=self.clients_per_round, replace=False)
            rnd_idx = idxs_users

        elif self.algo == 'rand':
            # Step 1: select 'm' clients with probability proportional to their dataset size with replacement
            idxs_users = np.random.choice(self.num_clients, p=self.ratio, size=self.clients_per_round, replace=True)

        elif self.algo == 'randint':
            # 'rand' for intermittent client availability
            delete = 0.2
            if (rnd % 2) == 0:
                del_idx = np.random.choice(int(self.num_clients/2), size=int(delete*self.num_clients/2), replace=False)
                search_idx = np.delete(np.arange(0, self.num_clients/2), del_idx)
            else:
                del_idx = np.random.choice(np.arange(self.num_clients/2, self.num_clients), size=int(delete*self.num_clients/2), replace=False)
                search_idx = np.delete(np.arange(self.num_clients/2, self.num_clients), del_idx)

            modified_data_ratios = [self.ratio[int(i)] for i in search_idx]/sum([self.ratio[int(i)] for i in search_idx])
            idxs_users = np.random.choice(search_idx, p=modified_data_ratios, size=self.clients_per_round, replace=True)

        elif self.algo == 'pow-norm':
            # power-of-norm strategy

            # set model params one time
            self.set_params(self.global_parameters)

            # Step 1: select 'd' clients with probability proportional to their dataset size without replacement
            rnd_idx = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)

            # Step 2: sort the selected clients in descending order of their loss
            norms, losses = [], []
            for i in rnd_idx:
                _, _, norm_i = self.train(i, update=False)
                print(f'norm of client {i}: {norm_i}, loss: {client_loss[i]}')
                norms.append(norm_i)
                losses.append(client_loss[i])
            norms = np.array(norms)
            print(f'norms: {norms}')
            losses = np.array(losses)
            print(f'losses: {losses}')
            norms = (norms - min(norms))/(max(norms) - min(norms))
            print(f'normalized norms: {norms}')
            losses = (losses - min(losses))/(max(losses) - min(losses))
            print(f'normalized losses: {losses}')
            print(f'norms+losses: {norms+losses}')
            repval = list(zip(norms+losses, rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))

            # Step 3: select indices of top 'm' clients from the sorted list
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo in ['pow-d', 'cpow-d', 'adapow-d']:
            # standard power-of-choice strategy

            # Step 1: select 'd' clients with probability proportional to their dataset size without replacement
            rnd_idx = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)

            # Step 2: sort the selected clients in descending order of their loss
            repval = list(zip([client_loss[i] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))

            # Step 3: select indices of top 'm' clients from the sorted list
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'rpow-d':
            # computation/communication efficient variant of 'pow-d'

            # Step 1: select 'd' clients with probability proportional to their dataset size without replacement
            rnd_idx1 = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)

            # Step 2: sort the selected clients in descending order of their proxy-loss
            repval = list(zip([client_loss_proxy[i] for i in rnd_idx1], rnd_idx1))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))

            # Step 3: select indices of top 'm' clients from the sorted list
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'pow-dint':  # TODO: whether to include cpow-dint and adapow-dint???
            # 'pow-d' for intermittent client availability
            delete = 0.2
            if (rnd % 2) == 0:
                del_idx = np.random.choice(int(self.num_clients/2), size=int(delete*self.num_clients/2), replace=False)
                search_idx = list(np.delete(np.arange(0, self.num_clients/2), del_idx))
            else:
                del_idx = np.random.choice(np.arange(self.num_clients/2, self.num_clients), size=int(delete*self.num_clients/2), replace=False)
                search_idx = list(np.delete(np.arange(self.num_clients/2, self.num_clients), del_idx))

            modified_data_ratios = [self.ratio[int(i)] for i in search_idx]/sum([self.ratio[int(i)] for i in search_idx])
            rnd_idx = np.random.choice(search_idx, p=modified_data_ratios, size=self.powd, replace=False)

            repval = list(zip([client_loss[int(i)] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'rpow-dint':
            # 'rpow-d' for intermittent client availability
            delete = 0.2
            if (rnd % 2) == 0:
                del_idx = np.random.choice(int(self.num_clients/2), size=int(delete*self.num_clients/2), replace=False)
                search_idx = list(np.delete(np.arange(0, self.num_clients/2), del_idx))
            else:
                del_idx = np.random.choice(np.arange(self.num_clients/2, self.num_clients), size=int(delete*self.num_clients/2), replace=False)
                search_idx = list(np.delete(np.arange(self.num_clients/2, self.num_clients), del_idx))

            modified_data_ratios = [self.ratio[int(i)] for i in search_idx]/sum([self.ratio[int(i)] for i in search_idx])
            rnd_idx = np.random.choice(search_idx, p=modified_data_ratios, size=self.powd, replace=False)

            repval = list(zip([client_loss_proxy[int(i)] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'afl':
            # benchmark strategy
            soft_temp = 0.01
            sorted_loss_idx = np.argsort(client_loss_proxy)

            for j in sorted_loss_idx[:int(self.delete_ratio*self.num_clients)]:
                client_loss_proxy[j]=-np.inf

            loss_prob = np.exp(soft_temp*client_loss_proxy)/sum(np.exp(soft_temp*client_loss_proxy))
            idx1 = np.random.choice(int(self.num_clients), p=loss_prob, size = int(np.floor((1-self.rnd_ratio)*self.clients_per_round)),
                                    replace=False)

            new_idx = np.delete(np.arange(0,self.num_clients),idx1)
            idx2 = np.random.choice(new_idx, size = int(self.clients_per_round-np.floor((1-self.rnd_ratio)*self.clients_per_round)), replace=False)

            idxs_users = list(idx1)+list(idx2)

        return idxs_users, rnd_idx