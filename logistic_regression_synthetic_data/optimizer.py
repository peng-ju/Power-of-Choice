from utils import read_data
import numpy as np

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1).reshape(ex.shape[0],1)
    return ex/sum_ex

class FederatedOptimizer(object):
    def __init__(self, lr, bs, le, seltype, powd, train_data_dir, test_data_dir, sample_ratio, num_classes=10):
        # hyperparameters
        self.lr = lr  # learning rate
        self.bs = bs  # batch size
        self.le = le  # local epochs
        self.seltype = seltype  # selection type
        self.powd = powd  # d (power of choice param)
        self.sample_ratio = sample_ratio  # number of clients to select per round, m = CK
        self.num_classes = num_classes
        self.local_losses = []  #
        # self.iter = 0
        # self.print_flg = True

        # read data
        _, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
        self.num_clients = len(self.train_data.keys())  # number of clients, K
        self.ratio = self.get_ratio()  # ratio, p_k for each client k
        # print("num_clients: ", self.num_clients)
        self.dim = np.array(self.train_data['f_00000']['x']).shape[1]  # size of feature vector
        self.central_parameter = np.zeros((self.dim, num_classes))  # size(dim, 10) ??
        # self.init_central = self.central_parameter + 0  # ??
        # self.local_parameters = np.zeros([self.num_clients, self.dim])  # params for each client


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

    def loss(self, X, y):
        """ compute loss for logistic model given data X and labels y """
        w = self.central_parameter
        y_hat = np.zeros((len(y), self.num_classes))
        y_hat[np.arange(len(y)),y.astype('int')] = 1
        loss = - np.sum(y_hat * np.log(softmax(X@w)))/X.shape[0]
        
        return loss

    def compute_gradient(self, x, i):
        """ compute gradient for logistic model  given data X and labels y """
        uname = 'f_{0:05d}'.format(i) 
        A = np.array(self.train_data[uname]['x'])
        y = np.array(self.train_data[uname]['y'])

        sample_idx = np.random.choice(A.shape[0], self.bs)
        a = A[sample_idx]
        targets = np.zeros((self.bs, 10))  # one-hot encoding for 10 classes
        targets[np.arange(self.bs), y[sample_idx].astype('int')] = 1  # same as? targets[:, y[sample_idx].astype('int')] = 1

        grad = - a.T @ (targets - softmax(a @ x))/self.bs  # https://medium.com/analytics-vidhya/ml-from-scratch-logistic-regression-gradient-descent-63b6beb1664c
        grad[:61] += 10e-4 * self.central_parameter[:61]  # ?? l2 regularization

        return grad

    def evaluate(self):
        glob_losses, local_losses = [], []
        for i in range(self.num_clients):
            uname = 'f_{0:05d}'.format(i) 
            A = np.array(self.train_data[uname]['x'])
            y = np.array(self.train_data[uname]['y'])
            glob_losses.append(self.loss(A, y) * self.ratio[i])
            local_losses.append(self.loss(A, y))

        glob_losses = np.array(glob_losses)

        return np.sum(glob_losses), local_losses

    def select_client(self, loc_loss):
        assert len(loc_loss) in [0, self.num_clients], "juups, losses were computed over all clients: incorrect assertion"
        # print("len(loc_loss): ", len(loc_loss))
        if not loc_loss:
            idxs_users = np.random.choice(self.num_clients, size=self.sample_ratio, replace=False)

        else:
            if self.seltype == 'rand':
                idxs_users = np.random.choice(self.num_clients, p=self.ratio, size=self.sample_ratio, replace=True)

            elif self.seltype == 'pow-d' or self.seltype == 'adapow-d':
                rnd_idx = np.random.choice(self.num_clients, p=self.ratio, size=self.powd, replace=False)
                repval = list(zip([loc_loss[i] for i in rnd_idx], rnd_idx))
                repval.sort(key=lambda x: x[0], reverse=True)
                rep = list(zip(*repval))
                idxs_users = rep[1][:int(self.sample_ratio)]

        return idxs_users

