import numpy as np
from optimizer import *

class FedAvg(FederatedOptimizer):
    def __init__(self, *args, **kwargs):
        # run base class constructor
        # print(*args, **kwargs)
        super(FedAvg, self).__init__(*args, **kwargs)
        # self.iter = 0

    def local_update(self, local_losses):
        # self.iter += 1
        # if self.seltype == "adapow-d" and self.iter >= 400:
        #     self.powd = self.sample_ratio
            # print(self.powd)

        # assert self.powd in [2*self.sample_ratio, 10*self.sample_ratio, self.sample_ratio], "oops"
        # if self.seltype == 'adapow-d' and self.iter >= 400:
        #     assert self.powd == self.sample_ratio, f"if - oops, {self.powd, self.iter, self.sample_ratio}"
        # elif self.seltype == 'adapow-d' and self.iter < 400:
        #     assert self.powd == 30, "else - oops"

        # find the set of active clients
        active_clients = self.select_client(local_losses)
        # print(len(local_losses))
        # assert len(local_losses) in [0, self.num_clients], "whoops, losses computed over all clients"

        delta = list()
        weight = 1 / self.sample_ratio
        # assert weight == 1/self.sample_ratio, "oops"

        # for each client
        for i in active_clients:
            lr = self.lr
            local_parameters = self.central_parameter + 0  # copy central parameters ??
            # for each local epoch of SGD
            for t in range(self.le):
                scale = 1
                local_parameters -= lr * self.compute_gradient(local_parameters, i)*scale
            delta.append((local_parameters - self.central_parameter)*weight)

        delta = np.array(delta)
        
        return delta, active_clients

    def aggregate(self, delta):
        self.central_parameter += np.sum(delta, axis=0)


