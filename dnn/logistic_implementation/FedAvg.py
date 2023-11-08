import logging
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils import read_data, SyntheticDataset

class FedAvg(object):
    def __init__(self, lr, bs, localE, algo, powd, num_clients, clients_per_round, train_data_dir, test_data_dir, num_classes, device=None):
        """ initialize federated optimizer """
        # hyperparameters
        self.lr = lr  # learning rate
        self.bs = bs  # batch size
        self.localE = localE  # local epochs
        self.algo = algo  # client selection algorithm
        self.powd = powd  # d (power of choice param)
        self.clients_per_round = clients_per_round  # clients per round, m
        self.num_classes = num_classes  # number of classes in the dataset
        self.device = device

        # read data
        self.cnt = 0
        _, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)

        # custom
        self.trainset = SyntheticDataset(train_data_dir, train=True)
        self.testset = SyntheticDataset(test_data_dir, train=False)

        self.num_clients = num_clients  # len(self.train_data.keys())  # number of clients, K
        self.ratio = self.get_ratio()  # ratio, p_k for each client k
        self.dim = np.array(self.train_data['f_00000']['x']).shape[1]  # input dimension
        
        # defining the model: here, logistic regression
        self.model = torch.nn.Linear(self.dim, self.num_classes, bias=False)
        
        # defining loss function: here, cross-entropy
        self.criterion = torch.nn.CrossEntropyLoss()

        # defining the optimizer: here, SGD
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=10e-4)

        # intial global params for server (similar behavior if initialized to random values)
        self.global_parameters = []
        with torch.no_grad():
            for param in self.model.parameters():
                self.global_parameters.append(torch.zeros_like(param))


    def set_params(self, parameters):
        """ set parameters of global model """
        with torch.no_grad():
            for model_param, param in zip(self.model.parameters(), parameters):
                model_param.copy_(param)
    

    def get_params(self):
        """ get parameters of global model """
        local_parameters = []
        with torch.no_grad():
            for param in self.model.parameters():
                local_parameters.append(param.detach().clone())
        return local_parameters


    def get_ratio(self):
        """ given the datasize of each client, compute the ratio p_k = |D_k|/|D| """
        total_size = 0
        ratios = np.zeros(self.num_clients)
        for i in range(self.num_clients):
            key = 'f_{0:05d}'.format(i)
            local_size = np.array(self.train_data[key]['x']).shape[0]
            ratios[i] = local_size
            total_size += local_size

        return ratios/total_size

    def eval(self, i, on_data='test'):
        """ compute loss, acc for client `i` on train/test data """
        self.model.eval()

        # fetch data for client `i`
        uname = 'f_{0:05d}'.format(i)
        dataset = self.testset if on_data == 'test' else self.trainset
        datasubset = Subset(dataset, indices=dataset.partitions[uname])
        dataloader = DataLoader(datasubset,
                                batch_size=len(datasubset), #self.bs,
                                shuffle=True,
                                pin_memory=True)

        loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # foward pass
                outputs = self.model(data)
                # torch.nn.functional.softmax(outputs) == softmax(X@w)

                # compute loss
                loss_tmp = self.criterion(outputs, target)
                loss += loss_tmp.item() * data.size(0)

                # prediction
                _, pred_labels = torch.max(outputs,1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum((pred_labels == target).float()).item()

                total += data.size(0)
        loss /= total
        acc = correct/total
        return loss, acc

    def evaluate(self, on_data):
        """ evaluate global loss and local losses for all clients

        local losses: loss for each client
        global loss: average of all local client losses weighted by ratio p_k
        """
        global_loss = 0
        global_acc = 0
        local_losses = []
        client_comptime = []
        local_acc = []

        # compute loss for each client
        for i in range(self.num_clients):
            comptime_start = time.time()
            loss_i, acc_i = self.eval(i, on_data)
            client_comptime.append(time.time() - comptime_start)
            global_loss += loss_i * self.ratio[i]
            global_acc += acc_i * self.ratio[i]
            local_losses.append(loss_i)
            local_acc.append(acc_i)

        # # trick
        # loss, correct, total = 0, 0, 0
        # dataset = self.testset if on_data == 'test' else self.trainset
        # dataloader = DataLoader(dataset,
        #                         batch_size=min(self.bs, len(dataset)),
        #                         shuffle=True,
        #                         pin_memory=True)
        # self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.bs, len(self.trainset)), shuffle=True)
        # with torch.no_grad():
        #     for batch_idx, (data, target) in enumerate(dataloader):
        #         data, target = data.to(self.device), target.to(self.device)
                
        #         # foward pass
        #         outputs = self.model(data)
        #         # torch.nn.functional.softmax(outputs) == softmax(X@w)

        #         # compute loss
        #         loss_tmp = self.criterion(outputs, target)
        #         loss += loss_tmp.item() * data.size(0)

        #         # prediction
        #         _, pred_labels = torch.max(outputs,1)
        #         pred_labels = pred_labels.view(-1)
        #         correct += torch.sum((pred_labels == target).float()).item()

        #         total += data.size(0)
        # loss /= total
        # acc = correct/total
        # # assert np.allclose(loss, global_loss), f'losses are not equal, {loss=}, {global_loss=}'

        return global_loss, global_acc, local_losses, local_acc, client_comptime

    def train(self, i):
        """ compute loss, acc for client `i` on train data and run optimizer step """
        self.model.train()

        # fetch single mini-batch for client `i` (stochasticity)
        uname = 'f_{0:05d}'.format(i) 
        datasubset = Subset(self.trainset, indices=self.trainset.partitions[uname])
        dataloader = DataLoader(datasubset,
                                batch_size=min(self.bs, len(datasubset)),
                                shuffle=True,
                                pin_memory=True)
        data, target = next(iter(dataloader))

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass
        outputs = self.model(data)

        # compute loss
        loss = self.criterion(outputs, target)

        # prediction
        _, pred_labels = torch.max(outputs,1)
        pred_labels = pred_labels.view(-1)
        acc = torch.mean((pred_labels == target).float())

        # backward pass - compute gradients
        loss.backward()

        # backward pass - update weights
        self.optimizer.step()

        return loss.item(), acc.item()

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
                tmp_loss, tmp_acc = self.train(i)
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
            
            # average local parameters
            for global_param in self.global_parameters:
                global_param.div_(len(client_params))


    def select_clients(self, client_loss, client_loss_proxy, args, rnd):
        '''
        Client selection part returning the indices the set $\mathcal{S}$ and $\mathcal{A}$
        Assumes that we have the list of local loss values for ALL clients

        :param data_ratios: $p_k$
        :param cli_loss: actual local loss F_k(w)
        :param cli_val: proxy of the local loss
        :param args: variable arguments
        :param rnd: communication round index
        :return: idxs_users (indices of $\mathcal{S}$), rnd_idx (indices of $\mathcal{A}$)
        '''
        rnd_idx = []
        if client_loss == []:
            # For the first round, select 'm' clients uniformly at random
            idxs_users = np.random.choice(self.num_clients, size=self.clients_per_round, replace=False)
            rnd_idx = idxs_users

        elif self.algo == 'rand':
            # Step 1: select 'm' clients with probability proportional to their loss with replacement
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

        elif self.algo == 'pow-d' or self.algo == 'adapow-d':
            # standard power-of-choice strategy

            # Step 1: select 'd' clients with probability proportional to their loss without replacement
            rnd_idx = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)

            # Step 2: sort the selected clients in descending order of their loss
            repval = list(zip([client_loss[i] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))

            # Step 3: select indices of top 'm' clients from the sorted list
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'rpow-d':
            # computation/communication efficient variant of 'pow-d'

            # Step 1: select 'd' clients with probability proportional to their loss without replacement
            rnd_idx1 = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)

            # Step 2: sort the selected clients in descending order of their proxy-loss
            repval = list(zip([client_loss_proxy[i] for i in rnd_idx1], rnd_idx1))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))

            # Step 3: select indices of top 'm' clients from the sorted list
            idxs_users = rep[1][:int(self.clients_per_round)]

        elif self.algo == 'pow-dint':
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
            # TODO: args.____ needs to be changed to self.____
            soft_temp = 0.01
            sorted_loss_idx = np.argsort(client_loss_proxy)

            for j in sorted_loss_idx[:int(args.delete_ratio*args.num_clients)]:
                client_loss_proxy[j]=-np.inf

            loss_prob = np.exp(soft_temp*client_loss_proxy)/sum(np.exp(soft_temp*client_loss_proxy))
            idx1 = np.random.choice(int(args.num_clients), p=loss_prob, size = int(np.floor((1-args.rnd_ratio)*args.clients_per_round)),
                                    replace=False)

            new_idx = np.delete(np.arange(0,args.num_clients),idx1)
            idx2 = np.random.choice(new_idx, size = int(args.clients_per_round-np.floor((1-args.rnd_ratio)*args.clients_per_round)), replace=False)

            idxs_users = list(idx1)+list(idx2)

        return idxs_users, rnd_idx