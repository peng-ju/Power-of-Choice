import numpy as np
from optimizer import *

class FedAvg(FederatedOptimizer):
    def __init__(self, *args, **kwargs):
        super(FedAvg, self).__init__(*args, **kwargs)

    def local_update(self, local_losses):
        """ logic for each communication round in Federated Learning """

        # find the set of active clients
        active_clients = self.select_client(local_losses)
        
        weights = list()
        # for each client
        for i in active_clients:
            # send global parameters to client `i`
            local_parameters = np.copy(self.global_parameter)
            
            # run E steps of SGD on client `i`
            for t in range(self.le):
                local_parameters -= self.lr * self.compute_gradient(local_parameters, i)

            # send local parameters from client `i` to server
            weights.append(local_parameters)

        weights = np.array(weights)
        return weights, active_clients

    def aggregate(self, weights):
        """ aggregation strategy in FedAvg """
        self.global_parameter = np.mean(weights, axis=0)


