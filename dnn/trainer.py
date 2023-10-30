import os
import time
import pathlib
import logging
import argparse
import numpy as np
import random

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from dist_optimizer import DistOptimizer
import utils
import models


logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
logging.debug("This message should appear on the console")

# define device
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def run(rank, args):
    # init logs directory
    save_path = "./logs/"
    fracC = args.clients_per_round/args.num_clients
    fold = f"lr{args.lr:.4f}_bs{args.bs}_cp{args.localE}_a{args.alpha:.2f}_e{args.seed}_r0_n{args.num_clients}_f{fracC:.2f}/"
    if args.commE:
        fold = "com_"+fold
    folder_name = save_path + args.name + "/" + fold
    file_name = f"{args.algo}_rr{args.rnd_ratio:.2f}_dr{args.delete_ratio:.2f}_lr{args.lr:.3f}_bs{args.bs:d}_cp{args.localE:d}"\
                    f"_a{args.alpha:.2f}_e{args.seed}_r{rank}_n{args.num_clients}_f{fracC:.2f}_p{args.powd}.csv"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # initiate log file
    saveFileName = folder_name+file_name
    args.out_fname = saveFileName
    args_str = [f"{key},{value}" for (key, value) in vars(args).items()]
    with open(args.out_fname, "w+") as f:
        print("BEGIN-TRAINING\n" + "\n".join(args_str) + \
            "\nEpoch,itr,loss,trainloss,avg:Loss,Prec@1,avg:Prec@1,val,trainval,updtime,comptime,seltime,entime", file=f)

    # seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    partitioner, dataratios, train_loader, test_loader = utils.partition_dataset(args, rnd=0)

    # tracking client loss values, frequency for each client
    client_freq, client_loss_proxy = np.zeros(args.num_clients), np.zeros(args.num_clients)

    # define model
    input_dims = np.prod(args.img_size)
    if args.model == "MLP":
        model = models.MLP_FMNIST(dim_in=input_dims, dim_hidden1=64, dim_hidden2 = 30, dim_out=args.num_classes).to(device)
    elif args.model == "CNN":
        model = models.CNN_CIFAR(args).to(device)

    # allocate buffer for global and aggregate parameters
    # ref: https://discuss.pytorch.org/t/how-to-assign-an-arbitrary-tensor-to-models-parameter/44082/3
    global_parameters = []
    aggregate_parameters = []
    with torch.no_grad():
        for param in model.parameters():
            global_parameters.append(param.detach().clone())
            aggregate_parameters.append(torch.zeros_like(param))            

    # define loss function
    criterion = nn.NLLLoss().to(device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=args.momentum, 
                                nesterov=False,
                                weight_decay=1e-4)
    # optimizer = DistOptimizer(model.parameters(),
    #                             lr=args.lr,
    #                             gmf=args.gmf, # set to 0
    #                             mu = args.mu, # set to 0
    #                             ratio=dataratios[rank],
    #                             momentum=args.momentum, # set to 0
    #                             nesterov = False,
    #                             weight_decay=1e-4)

    # randomly select clients for the first round
    replace_param = False
    if args.algo =="rand":
        replace_param = True
    idxs_users = np.random.choice(args.num_clients, size=args.clients_per_round, replace=replace_param)

    # start communication rounds
    for rnd in range(args.rounds):
        round_start = time.time()

        # (optional) decay learning rate according to round index
        if args.decay == True:
            # update_learning_rate(optimizer, rnd, args.lr)
            if rnd == 149:
                lr = args.lr/2
                logging.info("Updating learning rate to {}".format(lr))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if rnd == 299:
                lr = args.lr/4
                logging.info("Updating learning rate to {}".format(lr))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        # zero aggregate parameters for accumulation of local parameters
        with torch.no_grad():
            for param in aggregate_parameters:
                param.zero_()

        # for each client `i`
        for i in idxs_users:
            # send global parameters to client `i`
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_parameters):
                    param.copy_(global_param)
            
            # run E steps of SGD on client `i`
            loss_final = 0
            comm_update_start = time.time()
            for t in range(args.localE):
                singlebatch_loader = utils.partitiondata_loader(partitioner, i, args.bs)
                loss = train(i, model, criterion, optimizer, singlebatch_loader, t)
                loss_final += loss/args.localE
            comm_update_end = time.time()
            update_time = comm_update_end - comm_update_start

            # send local parameters from client `i` to server for aggregation
            with torch.no_grad():
                weight = 1/args.clients_per_round
                for aggregate_param, param in zip(aggregate_parameters, model.parameters()):
                    aggregate_param.add_(param, alpha=weight)
            
            # update client frequency and loss values
            client_freq[i] += 1
            client_loss_proxy[i] = loss_final

        # (??) getting value function for client selection (required only for "rpow-d", "afl")
        not_visited = np.where(client_freq == 0)[0]
        for j in not_visited:
            if args.algo == "afl":
                client_loss_proxy[j] = -np.inf
            else:
                client_loss_proxy[j] = np.inf

        # update global parameters
        with torch.no_grad():
            for global_param, aggregate_param in zip(global_parameters, aggregate_parameters):
                global_param.copy_(aggregate_param)

        # set model with global parameters
        with torch.no_grad():
            for param, global_param in zip(model.parameters(), global_parameters):
                param.copy_(global_param)

        # evaluate test accuracy
        test_acc, test_loss = evaluate(model, test_loader, criterion)

        # evaluate loss values and sync selected frequency
        client_loss, client_comptime = evaluate_clients(model, criterion, partitioner)
        train_loss = sum([client_loss[i]*dataratios[i] for i in range(args.num_clients)])
        train_loss1 = sum(client_loss)/args.num_clients

        # select clients for the next round
        sel_time, comp_time = 0, 0
        sel_time_start = time.time()
        idxs_users, rnd_idx = utils.select_clients(dataratios, client_loss, client_loss_proxy, args, rnd)
        # print(f"len rnd_idx {len(rnd_idx)} idxs_users {len(idxs_users)}")
        sel_time_end = time.time()
        sel_time = sel_time_end - sel_time_start

        if args.algo == "pow-d" or args.algo == "pow-dint":
            comp_time = max([client_comptime[int(i)] for i in rnd_idx])

        # record metrics
        round_end = time.time()
        round_duration = round(round_end - round_start, 1)
        logging.info(f"[{round_duration} s] Round {rnd} rank {rank} test accuracy {test_acc:.3f} test loss {test_loss:.3f}")
        with open(args.out_fname, "+a") as f:
            print(f"{rnd},{-1},{test_loss:.4f},{train_loss:.4f},-1,-1,-1,{test_acc:.4f},{train_loss1:.4f},"
                  f"{update_time:.4f},{comp_time:.4f},{sel_time:.4f},{update_time+comp_time+sel_time:.4f}", file=f)
    return


def evaluate_clients(model, criterion, partition):
    """
    Evaluate each client on their local train dataset against the current global model

    Evaluating each client"s local loss values for the current global model for client selection
    :param model: current global model
    :param criterion: loss function
    :param partition: dataset dict for clients
    :return: cli_loss = list of local loss values, cli_comptime = list of computation time
    """

    client_comptime, client_loss = [], []
    model.eval()

    # Get data from client to evaluate local loss on
    for i in range(args.num_clients):
        partitioned = partition.use(i)

        # cpow-d
        if args.commE:
            # single batch loader
            seldata_idx = random.sample(range(len(partitioned)), k=int(min(args.bs, len(partitioned))))
            partitioned = torch.utils.data.Subset(partitioned, indices=seldata_idx)

        train_loader = torch.utils.data.DataLoader(partitioned,
                                                   batch_size=len(partitioned),
                                                   shuffle=False,
                                                   pin_memory=False,
                                                   num_workers=0)

        # Compute local loss values or proxies for the clients
        tmp, total = 0,0
        with torch.no_grad():
            comptime_start = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                outputs = model(data)
                loss = criterion(outputs, target)
                tmp += loss.item()
                total += 1
            final_loss = tmp/total
            comptime_end = time.time()
            client_comptime.append(comptime_end-comptime_start)
            client_loss.append(final_loss)

    return client_loss, client_comptime


def evaluate(model, test_loader, criterion):
    """
    Evaluate test accuracy
    Evaluate model on full test dataset
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # Get test accuracy for the current model
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)

            # Inference
            outputs = model(data)
            batch_loss = criterion(outputs,target)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels.view(-1), target)).item() / len(pred_labels)
            total += 1

        acc = (correct / total) * 100
        los = loss/total

    return acc, los


def train(client_idx, model, criterion, optimizer, loader, epoch):
    """
    train model on the sampled mini-batch for $\tau$ epochs
    """

    model.train()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        data = data.to(device, non_blocking = True)
        target = target.to(device, non_blocking = True)

        # forward pass
        output = model(data)
        batch_loss = criterion(output, target)

        # backward pass
        batch_loss.backward()

        # gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # write log files
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(output, 1)
        correct += torch.sum(torch.eq(pred_labels.view(-1), target)).item()/len(pred_labels)
        total += 1

        acc = (correct / total)*100
        los = loss / total

        if batch_idx % args.print_freq == 0 and args.save:
            logging.debug("epoch {} itr {}, "
                         "client_idx {}, loss value {:.4f}, train accuracy {:.3f}"
                         .format(epoch, batch_idx, client_idx, los, acc))

            with open(args.out_fname, "+a") as f:
                print(f"{epoch},{batch_idx},{los:.4f},-1,-1,"
                      f"{acc:.3f},-1,-1,-1,-1,-1,-1", file=f)

    with open(args.out_fname, "+a") as f:
        print(f"{epoch},{batch_idx},{los:.4f},-1,-1,"
              f"{acc:.3f},-1,-1,-1,-1,-1,-1", file=f)

    return los


def init_processes(rank, size, world_size, fn):
    """ Initialize the distributed environment. """
    
    print(f"rank {rank} size {size}")
    dist.init_process_group(backend=args.backend, 
                            init_method=args.initmethod, 
                            rank=rank, 
                            world_size=world_size)
    fn(rank, size)


if __name__ == "__main__":
    args = args_parser()
    rank = args.rank
    # clients_per_round = args.clients_per_round
    # world_size = args.world_size

    # mp.spawn(init_processes, args=(size, world_size, run), nprocs=world_size, join=True)
    # # init_processes(rank, size, run)
    run(rank, args)
