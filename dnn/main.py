import os
import json
import time
import random
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm

import matplotlib.cm as cm
c_t = cm.get_cmap('tab10')

from FedAvg import FedAvg
from plot import make_plot


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
    # parser.add_argument("--constantE", action="store_true", help="whether all the local workers have an identical number of local epochs or not")  # TODO: add functionality
    parser.add_argument("--bs", default=64, type=int, help="batch size on each worker/client, $b$")
    parser.add_argument("--lr", default=0.1, type=float, help="client learning rate, $\eta$")
    parser.add_argument('--decay', nargs='*', type=int, help='rounds to decay LR')
    parser.add_argument("--alpha", default=0.2, type=float, help="control the non-iidness of dataset")
    parser.add_argument("--NIID", action="store_true", help="whether the dataset is non-iid or not")
    # parser.add_argument("--p", "-p", action="store_true", help="whether the dataset is partitioned or not")

    # algo specific hyperparams
    parser.add_argument("--powd", default=6, type=int, help="number of selected subset workers per round ($d$)")
    # parser.add_argument("--momentum", default=0.0, type=float, help="local (client) momentum factor")
    # parser.add_argument("--mu", default=0, type=float, help="mu parameter in fedprox")
    # parser.add_argument("--gmf", default=0, type=float, help="global (server) momentum factor")
    parser.add_argument("--rnd_ratio", default=0.1, type=float, help="hyperparameter for afl")
    parser.add_argument("--delete_ratio", default=0.75, type=float, help="hyperparameter for afl")

    # logistics
    # parser.add_argument("--print_freq", default=100, type=int, help="print info frequency")
    # parser.add_argument("--num_workers", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--config", default=None, type=str, help="config file path")
    # parser.add_argument("--save", "-s", action="store_true", help="whether save the training results")
    
    # distributed setup  # TODO: add functionality
    # parser.add_argument("--backend", default="gloo", type=str, help="backend name")
    # parser.add_argument("--world_size", default=1, type=int, help="world size for distributed training")
    # parser.add_argument("--rank", default=0, type=int, help="the rank of worker")
    # parser.add_argument("--initmethod", default="tcp://", type=str, help="init method")

    args = parser.parse_args()
    return args


def run(rank, args):
    # init logs directory
    save_path = "./logs/"
    fracC = args.clients_per_round/args.num_clients
    fold = f"lr{args.lr:.4f}_bs{args.bs}_cp{args.localE}_a{args.alpha:.2f}_e{args.seed}_r0_n{args.num_clients:04d}_f{fracC:.2f}/"
    folder_name = save_path + args.name + "/" + fold
    file_name = f"{args.algo}_rr{args.rnd_ratio:.2f}_dr{args.delete_ratio:.2f}_p{args.powd:04d}_r{rank}.csv"
    os.makedirs(folder_name, exist_ok=True)

    # initiate log file
    args.out_fname = folder_name + file_name
    args_str = [f"{key},{value}" for (key, value) in vars(args).items()]
    with open(args.out_fname, "w+") as f:
        print("BEGIN-TRAINING\n" + "\n".join(args_str) + "\n" \
            "rank,round,epoch,test_loss,train_loss,test_acc,train_acc", file=f)

    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
    logging.info("This message should appear on the console")
    logging.info("Args: %s", args)

    # set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # run federated learning experiment for given configuration
    server = FedAvg(args.lr, args.bs, args.localE, args.algo, args.model, 
                    args.powd, args.num_clients, args.clients_per_round, 
                    args.dataset, args.num_classes, args.NIID, args.alpha, 
                    args.delete_ratio, args.rnd_ratio, args.seed, args.device)
    
    client_train_losses = []

    # tracking client loss values, frequency for each client
    client_freq, client_loss_proxy = np.zeros(args.num_clients), np.zeros(args.num_clients)

    for rnd in tqdm(range(args.rounds), desc=args.key):
        round_start = time.time()

        # (optional) decay learning rate according to round index
        if args.decay and rnd in args.decay:
            for param_group in server.optimizer.param_groups:
                param_group["lr"] /= 2
                    
        # reduce powd from K to m after half rounds (only for 'adapow-d')
        if args.algo == 'adapow-d' and rnd == args.rounds//2:
            server.powd = args.clients_per_round

        # find the set of active clients
        active_clients, rnd_idx = server.select_clients(client_train_losses, client_loss_proxy, rnd)

        # train active clients locally
        weights, losses, comm_update_times = server.local_update(active_clients)

        # update global parameter by aggregating weights
        server.aggregate(weights)

        # send global parameters to all clients for evaluation
        server.set_params(server.global_parameters)

        # evaluation
        train_loss, train_acc, client_train_losses, client_train_accs, client_comp_times_train = server.evaluate()
        test_loss, test_acc = server.evaluate_approx()

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
            logging.info(f"[{round_duration} s] Round {rnd} rank {rank} test accuracy {test_acc:.3f} test loss {train_loss:.3f}"
                         f" train accuracy {train_acc:.3f} train loss {train_loss:.3f} ")

        with open(args.out_fname, "+a") as f:
            # round,epoch,test_loss,train_loss,test_acc,train_acc
            print(f"{rank},{rnd},{-1},{test_loss:.4f},{train_loss:.4f},{test_acc:.4f},{train_acc:.4f}", file=f)
    
    print(f"Saving logs to {args.out_fname}")
    return args.out_fname

if __name__ == '__main__':
    ## parse command line arguments
    args = args_parser()

    ## define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    args.device = device

    ## define a key for the experiment
    args.key = "default"

    ## run from config, if provided
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file '{args.config}' does not exist!")
        
        with open(args.config) as f:
            config = json.load(f)

        # set common vars
        for key in config['common'].keys():
            setattr(args, key, config['common'][key])

        log_filenames = []
        for expt in config['different'].keys():
            args.key = expt
            # set different vars
            for key in config['different'][expt].keys():
                setattr(args, key, config['different'][expt][key])

            # run the configuration
            tmp_filename = run(0, args)
            log_filenames.append(tmp_filename)

        ## plot results
        for key in config['plots']:
            make_plot(log_filenames, key)
        
    ## run from command line arguments otherwise
    else:
        run(0, args)
