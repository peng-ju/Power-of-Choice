from utils import read_data
import numpy as np

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1).reshape(ex.shape[0],1)
    return ex/sum_ex

class FederatedOptimizer(object):
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
        self.dim = np.array(self.train_data['f_00000']['x']).shape[1]  # size of feature vector
        self.global_parameter = np.zeros((self.dim, self.num_classes))  # params for multi-class logistic regression

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

    def loss(self, w, i):
        """ compute loss for logistic model given weights `w` and data on client `i`
        Ref: https://medium.com/analytics-vidhya/ml-from-scratch-logistic-regression-gradient-descent-63b6beb1664c
        """
        # fetch data for client `i`
        uname = 'f_{0:05d}'.format(i) 
        X = np.array(self.train_data[uname]['x'])
        y = np.array(self.train_data[uname]['y'])
        n = num_samples = X.shape[0]
        assert num_samples == y.shape[0], "number of samples in X and y must match"

        # one-hot encoding for targets
        y_hat = np.zeros((n, self.num_classes))
        y_hat[np.arange(n), y.astype('int')] = 1

        # compute loss
        loss = - np.sum(y_hat * np.log(softmax(X@w)))/n
        
        return loss

    def evaluate(self):
        """ evaluate global and local losses for all clients """
        glob_losses, local_losses = [], []
        for i in range(self.num_clients):
            loss = self.loss(self.global_parameter, i)
            glob_losses.append(loss * self.ratio[i])
            local_losses.append(loss)

        return np.sum(glob_losses), local_losses
    
    def compute_gradient(self, w, i):
        """ compute gradient for logistic model  given weights `w` and data on client `i`
        Ref: https://medium.com/analytics-vidhya/ml-from-scratch-logistic-regression-gradient-descent-63b6beb1664c
        """
        # fetch data for client `i`
        uname = 'f_{0:05d}'.format(i) 
        X = np.array(self.train_data[uname]['x'])
        y = np.array(self.train_data[uname]['y'])

        # fetch mini-batch
        sample_idx = np.random.choice(X.shape[0], size=self.bs)
        x = X[sample_idx]

        # one-hot encoding for targets
        targets = np.zeros((self.bs, self.num_classes))
        targets[np.arange(self.bs), y[sample_idx].astype('int')] = 1  # one-hot encoding for 10 classes

        # compute gradient and apply l2-regularization
        grad = - x.T @ (targets - softmax(x @ w))/self.bs
        import pdb; pdb.set_trace()
        grad[:61] += 10e-4 * self.global_parameter[:61]  # l2-regularization

        return grad

    def select_client(self, loc_loss):
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

