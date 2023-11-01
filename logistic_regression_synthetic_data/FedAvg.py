import numpy as np
import torch

from utils import read_data

class FedAvg(object):
    def __init__(self, lr, bs, le, algo, powd, train_data_dir, test_data_dir, sample_ratio, num_classes=10):
        """ initialize federated optimizer """
        # hyperparameters
        self.lr = lr  # learning rate
        self.bs = bs  # batch size
        self.le = le  # local epochs
        self.algo = algo  # client selection algorithm
        self.powd = powd  # d (power of choice param)
        self.sample_ratio = sample_ratio  # clients per round, m
        self.num_classes = num_classes  # number of classes in the dataset

        # read data
        _, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
        self.num_clients = len(self.train_data.keys())  # number of clients, K
        self.ratio = self.get_ratio()  # ratio, p_k for each client k
        self.dim = np.array(self.train_data['f_00000']['x']).shape[1]  # input dimension

        # intial global params for server (similar behavior if initialized to random values)
        self.global_parameter = torch.zeros(self.num_classes, self.dim)
        
        # defining the model: here, logistic regression
        self.model = torch.nn.Linear(self.dim, self.num_classes, bias=False)
        
        # defining loss function: here, cross-entropy
        self.criterion = torch.nn.CrossEntropyLoss()

        # defining the optimizer: here, SGD
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=10e-4)

    def set_params(self, w):
        """ set global parameters of model """
        with torch.no_grad():
            self.model.weight.copy_(w)
    
    def get_params(self):
        """ get global parameters of model """
        return self.model.weight.detach().clone()

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

    def loss(self, i):
        """ compute loss for client `i` """
        # fetch data for client `i`
        uname = 'f_{0:05d}'.format(i) 
        X = torch.tensor(self.train_data[uname]['x'], dtype=torch.float32)
        y = torch.tensor(self.train_data[uname]['y'], dtype=torch.int64)

        # foward pass
        outputs = self.model(X)
        # torch.nn.functional.softmax(outputs) == softmax(X@w)

        # compute loss
        loss = self.criterion(outputs, y)

        return loss.item()

    def evaluate(self):
        """ evaluate global loss and local losses for all clients

        local losses: loss for each client
        global loss: average of all local client losses weighted by ratio p_k
        """
        global_loss = 0
        local_losses = []

        # send global parameters to all clients
        self.set_params(self.global_parameter)

        # compute loss for each client
        for i in range(self.num_clients):
            loss = self.loss(i)
            global_loss += loss * self.ratio[i]
            local_losses.append(loss)

        return global_loss, local_losses

    def sgd_step(self, i):
        """ run one step of SGD on client `i` and updates model parameters """
        # fetch data for client `i`
        uname = 'f_{0:05d}'.format(i) 
        X = torch.tensor(self.train_data[uname]['x'], dtype=torch.float32)
        y = torch.tensor(self.train_data[uname]['y'], dtype=torch.int64)

        # fetch mini-batch (stochasticity)
        sample_idx = np.random.choice(X.shape[0], size=self.bs)

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass
        outputs = self.model(X[sample_idx])

        # compute loss
        loss = self.criterion(outputs, y[sample_idx])

        # backward pass - compute gradients
        loss.backward()

        # backward pass - update weights
        self.optimizer.step()
        return

    def local_update(self, active_clients):
        """ train the set of active clients """
        weights = []
        for i in active_clients:
            # send global parameters to client `i`
            self.set_params(self.global_parameter)
            
            # run E steps of SGD on client `i`
            for _ in range(self.le):
                self.sgd_step(i)
            
            # get local parameters from client `i`
            local_parameters = self.get_params()
            weights.append(local_parameters)

        return torch.stack(weights), active_clients

    def aggregate(self, weights):
        """ aggregation strategy in FedAvg """
        self.global_parameter = torch.mean(weights, axis=0)


    def select_client(self, loc_loss):
        """ client selection strategy for each round of communication """
        if not loc_loss:
            # For the first round, select 'm' clients uniformly at random
            idxs_users = np.random.choice(self.num_clients, size=self.sample_ratio, replace=False)

        else:
            if self.algo == 'rand':
                # Step 1: select 'm' clients with probability proportional to their loss with replacement
                idxs_users = np.random.choice(self.num_clients, p=self.ratio, size=self.sample_ratio, replace=True)

            elif self.algo == 'pow-d' or self.algo == 'adapow-d':
                # Step 1: select 'd' clients with probability proportional to their loss without replacement
                rnd_idx = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)
                
                # Step 2: sort the selected clients in descending order of their loss
                repval = list(zip([loc_loss[i] for i in rnd_idx], rnd_idx))
                repval.sort(key=lambda x: x[0], reverse=True)
                rep = list(zip(*repval))
                
                # Step 3: select indices of top 'm' clients from the sorted list
                idxs_users = rep[1][:int(self.sample_ratio)]

        return idxs_users