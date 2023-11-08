import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
c_t = cm.get_cmap('tab10')

from FedAvg import FedAvg


def args_parser():
    """ parse command line arguments """

    # basic parameters
    parser = argparse.ArgumentParser(description="FMNIST baseline")
    parser.add_argument("--name", "-n", default="default", type=str, help="experiment name, used for saving results")
    parser.add_argument("--model", default="MLP", type=str, help="neural network model")
    parser.add_argument("--dataset", default="fmnist", type=str, help="type of dataset")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")

    # hyperparams
    parser.add_argument("--algo", default="rand", type=str, help="type of client selection ($\pi$)")
    parser.add_argument("--num_clients", default=100, type=int, help="total number of clients, $K$")
    parser.add_argument("--rounds", default=500, type=int, help="total communication rounds")
    parser.add_argument("--clients_per_round", default=1, type=int, help="number of local workers")
    parser.add_argument("--localE", default=30, type=int, help="number of local epochs, $E$")
    parser.add_argument("--constantE", action="store_true", help="whether all the local workers have an identical number of local epochs or not")
    parser.add_argument("--bs", default=64, type=int, help="batch size on each worker/client, $b$")
    parser.add_argument("--lr", default=0.1, type=float, help="client learning rate, $\eta$")
    parser.add_argument("--decay", default=1, type=bool, help="1: decay LR, 0: no decay")
    parser.add_argument("--alpha", default=0.2, type=float, help="control the non-iidness of dataset")
    parser.add_argument("--NIID", action="store_true", help="whether the dataset is non-iid or not")

    # algo specific hyperparams
    parser.add_argument("--powd", default=6, type=int, help="number of selected subset workers per round ($d$)")
    parser.add_argument("--commE", action="store_true", help="activation of $cpow-d$")
    parser.add_argument("--momentum", default=0.0, type=float, help="local (client) momentum factor")
    parser.add_argument("--mu", default=0, type=float, help="mu parameter in fedprox")
    parser.add_argument("--gmf", default=0, type=float, help="global (server) momentum factor")
    parser.add_argument("--rnd_ratio", default=0.1, type=float, help="hyperparameter for afl")
    parser.add_argument("--delete_ratio", default=0.75, type=float, help="hyperparameter for afl")

    # logistics
    parser.add_argument("--print_freq", default=100, type=int, help="print info frequency")
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--save", "-s", action="store_true", help="whether save the training results")
    parser.add_argument("--p", "-p", action="store_true", help="whether the dataset is partitioned or not")
    
    # distributed setup
    parser.add_argument("--backend", default="gloo", type=str, help="backend name")
    parser.add_argument("--world_size", default=1, type=int, help="world size for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="the rank of worker")
    parser.add_argument("--initmethod", default="tcp://", type=str, help="init method")

    args = parser.parse_args()
    return args

def make_plot(client_selection_type):
    ## plot settings
    # color maps reference: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    # line styles reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    ftsize = 16
    params = {'legend.fontsize': ftsize,
            'axes.labelsize': ftsize,
            'axes.titlesize':ftsize,
            'xtick.labelsize':ftsize,
            'ytick.labelsize':ftsize}
    plt.rcParams.update(params)
    lw = 2
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.figure(figsize=(16,14.5))
    plt.subplots_adjust(right=1.1, top=0.9)
    rcParams['axes.titlepad'] = 14

    for key in client_selection_type.keys():
        # fetch configuration
        algo, clients_per_round, powd, color, lstyle = client_selection_type[key]

        # load errors from json file
        with open(f'./logs/m={clients_per_round}_algo={key}_errors.json') as f:
            errors = json.load(f)
        
        # plot global loss for each configuration
        if algo =='rand' or algo =='adapow-d':
            p_label = algo
        else:
            p_label = algo+', d={}'.format(powd)
        plt.plot(errors, lw=lw, color=color, ls = lstyle, label=p_label)

    # update plot settings
    plt.ylabel('Global loss')
    plt.xlabel('Communication round')
    plt.xticks()
    plt.yticks()
    plt.legend(loc=1)
    plt.grid()
    plt.title('K=30, m={}'.format(clients_per_round))
    # plt.show()
    plt.savefig(f'synthetic_m={clients_per_round}.pdf', bbox_inches='tight')
    print(f'saving plot to synthetic_m={clients_per_round}.pdf')
    
    return

def run(args):
    ## create logs directory if not exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
    logging.info("This message should appear on the console")

    # set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # run federated learning experiment for given configuration
    server = FedAvg(args.lr, args.bs, args.localE, args.algo, args.powd, args.num_clients,
                    args.clients_per_round, train_data_dir, test_data_dir, args.num_classes, args.device)
    errors, local_losses_train = [], []

    # tracking client loss values, frequency for each client
    client_freq, client_loss_proxy = np.zeros(args.num_clients), np.zeros(args.num_clients)

    for rnd in tqdm(range(args.rounds), desc=key):
        round_start = time.time()

        # (optional) decay learning rate according to round index
        if args.decay == True:
            # update_learning_rate(optimizer, rnd, args.lr)
            if rnd == 300 or rnd == 600:
                for param_group in server.optimizer.param_groups:
                    param_group["lr"] /= 2
                    
        # reduce powd from K to m after half rounds (only for 'adapow-d')
        if args.algo == 'adapow-d' and rnd == args.rounds//2:
            server.powd = args.clients_per_round

        # find the set of active clients
        active_clients, rnd_idx = server.select_clients(local_losses_train, client_loss_proxy, args, rnd)

        # train active clients locally
        weights, losses, comm_update_times = server.local_update(active_clients)

        # update global parameter by aggregating weights
        server.aggregate(weights)

        # send global parameters to all clients for evaluation
        server.set_params(server.global_parameters)

        # evaluation
        global_loss_train, local_losses_train, local_accuracies_train, client_comp_times_train = server.evaluate('train')
        global_loss_test, local_losses_test, local_accuracies_test, client_comp_times_test = server.evaluate('test')
        errors.append(global_loss_train)

        ## bookkeeping
        # update client freq/loss values
        for i, loss_i in zip(active_clients, losses):
            client_freq[i] += 1
            client_loss_proxy[i] = loss_i

        # (??) getting value function for client selection (required only for "rpow-d", "afl")
        not_visited = np.where(client_freq == 0)[0]
        for j in not_visited:
            if args.algo == "afl":
                client_loss_proxy[j] = -np.inf
            else:
                client_loss_proxy[j] = np.inf
        
        # track max communication time
        if args.algo == "pow-d" or args.algo == "pow-dint":
            comp_time = max([client_comp_times_train[int(i)] for i in rnd_idx])

        # log results
        round_end = time.time()
        round_duration = round(round_end - round_start, 1)
        if round_duration > 1:
            logging.info(f"[{round_duration} s] Round {rnd} ") #rank {rank} test accuracy {test_acc:.3f} test loss {test_loss:.3f}")

    # save errors to json file
    with open(f'./logs/m={args.clients_per_round}_algo={key}_errors.json', 'w') as f:
        json.dump(errors, f)
        
    return 

if __name__ == '__main__':
    args = args_parser()

    ## hyperparameters
    train_data_dir = './synthetic_data/'
    test_data_dir = './synthetic_data/'
    args.lr = 0.05  # learning rate, \eta
    args.bs = 50  # batch size, b
    args.localE = 30  # local epoch, E
    args.rounds = 100  # total communication rounds, T/E
    args.clients_per_round = 1  # active clients per round, m
    args.num_clients = 30  # number of clients, K
    args.seed = 12345
    log_remote = False

    ## experiment configurations
    # key=experiment_id, value=(algo, powd, color, linestyle)
    client_selection_type = {
        'rand': ('rand', args.clients_per_round, 1, 'k', '-'),
        'powd2': ('pow-d', args.clients_per_round, args.clients_per_round*2, c_t(3), '-.'),
        'powd5': ('pow-d', args.clients_per_round, args.clients_per_round*10, c_t(0), '--'),
        # 'adapow30': ('adapow-d', args.clients_per_round, K, c_t(1), (0, (5, 10)))
    }

    # define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    args.device = device

    ## run experiments
    for key in client_selection_type.keys():
        # fetch configuration
        algo, clients_per_round, powd, color, lstyle = client_selection_type[key]

        args.algo = algo
        args.powd = powd
        run(args)

    make_plot(client_selection_type)