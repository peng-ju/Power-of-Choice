import os
import numpy as np
import random

import time
import pathlib
import logging

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from distoptim import fedavg
import util_v4 as util
import models
from params import args_parser


logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logging.debug('This message should appear on the console')

# define device
device = "cuda" if torch.cuda.is_available() else "cpu"

args = args_parser()

def run(rank, args):
    fracC = args.clients_per_round/args.num_clients
    print(fracC)
    # initiate experiments folder
    save_path = './logs/'  # '/users/name/'
    fold = 'lr{:.4f}_bs{}_cp{}_a{:.2f}_e{}_r0_n{}_f{:.2f}/'.format(args.lr, args.bs, args.localE, args.alpha, args.seed,
                                                                   args.num_clients, fracC)
    if args.commE:
        fold = 'com_'+fold
    folder_name = save_path+args.name+'/'+fold
    file_name = '{}_rr{:.2f}_dr{:.2f}_lr{:.3f}_bs{:d}_cp{:d}_a{:.2f}_e{}_r{}_n{}_f{:.2f}_p{}.csv'.format(args.seltype,
                                                    args.rnd_ratio, args.delete_ratio, args.lr, args.bs, args.localE,
                                                    args.alpha, args.seed, rank, args.num_clients, fracC, args.powd)
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # initiate log files
    saveFileName = folder_name+file_name
    args.out_fname = saveFileName
    with open(args.out_fname, 'w+') as f:
        print('BEGIN-TRAINING\n' 'World-Size,{ws}\n' 'Batch-Size,{bs}\n' 'Epoch,itr,'
            'loss,trainloss,avg:Loss,Prec@1,avg:Prec@1,val,trainval,updtime,comptime,seltime,entime'.format(
            ws=args.world_size, bs=args.bs), file=f)

    # seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    # partition, train_loader, test_loader, dataratios, datstat, endat = util.partition_dataset(size, args, 0)
    partitioner, dataratios, train_loader, test_loader = util.partition_dataset(args, 0)

    # initialization for client selection
    cli_loss, cli_freq, cli_val = np.zeros(args.num_clients)+1, np.zeros(args.num_clients), np.zeros(args.num_clients)

    # tmp_cli = [torch.tensor(0, dtype=torch.float32).to(device) for _ in range(args.size)]
    # tmp_cli = [0 for _ in range(args.clients_per_round)]
    # tmp_clifreq = [0 for _ in range(args.clients_per_round)]

    # dist.barrier()

    # randomly select clients for the first round
    # send = torch.zeros(args.size, dtype=torch.int32).to(device)
    # if rank == 0:
    #     replace_param = False
    #     if args.seltype =='rand':
    #         replace_param = True

    #     idxs_users = np.random.choice(args.num_clients, size=args.size, replace=replace_param)
    #     # send = [torch.tensor(int(ii)).to(device) for ii in idxs_users]
    #     send = torch.tensor(idxs_users).to(device)
    # dist.barrier()

    # dist.broadcast(tensor=send, src=0)
    # # for i in range(args.size):
    # #     dist.broadcast(tensor=send[i], src=0)
    # dist.barrier()
    # print('rank {}, send={}'.format(rank, send))
    # sel_idx = int(send[rank])

    # randomly select clients for the first round
    replace_param = False
    if args.seltype =='rand':
        replace_param = True
    idxs_users = np.random.choice(args.num_clients, size=args.clients_per_round, replace=replace_param)

    # define neural nets model
    # len_in = 1
    # for x in args.img_size:
    #     len_in *= x
    len_in = np.prod(args.img_size)
    # assert len_in == len_in1, 'len_in {} len_in1 {}'.format(len_in, len_in1)
    if args.model == 'MLP':
        model = models.MLP_FMNIST(dim_in=len_in, dim_hidden1=64, dim_hidden2 = 30, dim_out=args.num_classes).to(device)

    elif args.model == 'CNN':
        model = models.CNN_CIFAR(args).to(device)  # vgg

    # allocate buffer for global parameters
    # ref: https://discuss.pytorch.org/t/how-to-assign-an-arbitrary-tensor-to-models-parameter/44082/3
    global_parameters = []
    with torch.no_grad():
        for param in model.parameters():
            global_parameters.append(param.detach().clone())

    # allocate buffer for averaging local parameters
    cumulative_local_parameters = []
    with torch.no_grad():
        for param in model.parameters():
            cumulative_local_parameters.append(torch.zeros_like(param))

    # define criterion
    criterion = nn.NLLLoss().to(device)

    # defined optimizer
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=args.momentum, 
                                nesterov=False,
                                weight_decay=1e-4)
    # if args.optimizer == 'fedavg':
    #     optimizer = fedavg(model.parameters(),
    #                     lr=args.lr,
    #                     gmf=args.gmf, # set to 0
    #                     mu = args.mu, # set to 0
    #                     ratio=dataratios[rank],
    #                     momentum=args.momentum, # set to 0
    #                     nesterov = False,
    #                     weight_decay=1e-4)


    for rnd in range(args.rounds):
        round_start = time.time()
        # Initialize hyperparameters
        local_epochs = args.localE
        weight = 1/args.clients_per_round

        # Decay learning rate according to round index (optional)
        if args.decay == True:
            # update_learning_rate(optimizer, rnd, args.lr)
            if rnd == 149:
                lr = args.lr/2
                logging.info('Updating learning rate to {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if rnd == 299:
                lr = args.lr/4
                logging.info('Updating learning rate to {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # # Clients locally train for several local epochs
        # loss_final = 0
        # dist.barrier()
        # comm_update_start = time.time()
        # for t in range(local_epochs):
        #     singlebatch_loader = util.partitiondata_loader(partitioner, sel_idx, args.bs)
        #     loss = train(rank, model, criterion, optimizer, singlebatch_loader, t)
        #     loss_final += loss/local_epochs
        # dist.barrier()
        # comm_update_end = time.time()
        # update_time = comm_update_end - comm_update_start

        # execute one communication round
        with torch.no_grad():
            for param in cumulative_local_parameters:
                param.zero_()

        tmp_cli = []
        tmp_clifreq = []
        # for each client
        for i in idxs_users:
            tmp_clifreq.append(i)

            # send global parameters to client `i`
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_parameters):
                    param.copy_(global_param)
            
            # run E steps of SGD on client `i`
            loss_final = 0
            comm_update_start = time.time()
            for t in range(local_epochs):
                singlebatch_loader = util.partitiondata_loader(partitioner, i, args.bs)
                loss = train(i, model, criterion, optimizer, singlebatch_loader, t)
                loss_final += loss/local_epochs
            comm_update_end = time.time()
            update_time = comm_update_end - comm_update_start
            tmp_cli.append(loss_final)

            # send local parameters from client `i` to server
            # update global parameter by aggregating weights
            with torch.no_grad():
                for cumulative_param, param in zip(cumulative_local_parameters, model.parameters()):
                    cumulative_param.add_(param, alpha=weight)


        # Getting value function for client selection (required only for 'rpow-d', 'afl')
        # dist.barrier()      # TODO: implement with multi-arm bandit
        # dist.all_gather(tmp_cli, torch.tensor(loss_final).to(device))
        # dist.all_gather(tmp_clifreq, torch.tensor(int(sel_idx)).to(device))
        # dist.barrier()
        # for i, i_val in enumerate(tmp_clifreq):
        #     cli_freq[i_val.item()]+= 1         # Cli freq is the entire clients that are selected for all rounds
        #     cli_val[i_val.item()] = tmp_cli[i].item()
        # not_visited = np.where(cli_freq == 0)[0]

        # Getting value function for client selection (required only for 'rpow-d', 'afl')
        for loss_i, i in zip(tmp_cli, tmp_clifreq):
            cli_freq[i] += 1         # Cli freq is the entire clients that are selected for all rounds
            cli_val[i] = loss_i    # cli_val is the loss value of the selected clients
        not_visited = np.where(cli_freq == 0)[0]

        for ii in not_visited:
            if args.seltype == 'afl':
                cli_val[ii] = -np.inf
            else:
                cli_val[ii] = np.inf

        # synchronize parameters
        # dist.barrier()
        # optimizer.average(weight=weight)
        # dist.barrier()

        # synchronize parameters
        with torch.no_grad():
            for global_param, cumulative_param in zip(global_parameters, cumulative_local_parameters):
                global_param.copy_(cumulative_param)

        # set model with global parameters
        with torch.no_grad():
            for param, global_param in zip(model.parameters(), global_parameters):
                param.copy_(global_param)

        # evaluate test accuracy
        test_acc, test_loss = evaluate(model, test_loader, criterion)

        # evaluate loss values and sync selected frequency
        cli_loss, cli_comptime = evaluate_client(model, criterion, partitioner)
        train_loss = sum([cli_loss[i]*dataratios[i] for i in range(args.num_clients)])
        train_loss1 = sum(cli_loss)/args.num_clients

        # dist.barrier()
        # # Select client for each round, in total m ranks
        # send = torch.zeros(args.size, dtype=torch.int32).to(device)
        # # send1 = torch.zeros(args.size, dtype=torch.int32).to(device)
        # comp_time, sel_time = 0, 0

        # # master client runs selection algorithm and sends the selected client index to all other clients
        # if rank == 0:
        #     sel_time_start = time.time()
        #     idxs_users, rnd_idx = util.select_clients(dataratios, cli_loss, cli_val, args, rnd)
        #     print(f'len rnd_idx {len(rnd_idx)} idxs_users {len(idxs_users)}')
        #     sel_time_end = time.time()
        #     sel_time = sel_time_end - sel_time_start

        #     if args.seltype == 'pow-d' or args.seltype == 'pow-dint':
        #         comp_time = max([cli_comptime[int(i)] for i in rnd_idx])

        #     # send = [torch.tensor(int(ii)).to(device) for ii in idxs_users]
        #     send = torch.tensor(idxs_users).to(device)
        # dist.barrier()
        # dist.broadcast(tensor=send, src=0)
        # # for i in range(args.size):
        # #     dist.broadcast(tensor=send[i], src=0)
        # dist.barrier()
        # print('rank {}, send={}'.format(rank, send))
        # # print('rank {}, send1={}'.format(rank, send1))
        # # print(f'{send1.dtype=}, {(send1[0].dtype)=}')
        # # print(f'{(send[0].dtype)=}')
        # sel_idx = int(send[rank])  # selected client index

        # select clients for the next round
        sel_time, comp_time = 0, 0
        sel_time_start = time.time()
        idxs_users, rnd_idx = util.select_clients(dataratios, cli_loss, cli_val, args, rnd)
        # print(f'len rnd_idx {len(rnd_idx)} idxs_users {len(idxs_users)}')
        sel_time_end = time.time()
        sel_time = sel_time_end - sel_time_start

        if args.seltype == 'pow-d' or args.seltype == 'pow-dint':
            comp_time = max([cli_comptime[int(i)] for i in rnd_idx])

        # record metrics
        round_end = time.time()
        round_duration = round(round_end - round_start, 1)
        logging.info("[{} s] Round {} rank {} test accuracy {:.3f} test loss {:.3f}".format(round_duration, rnd, rank, test_acc, test_loss))
        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},{loss:.4f},{trainloss:.4f},{filler},'
                  '{filler},{filler},'
                  '{val:.4f},{other:.4f},{updtime:.4f},{comptime:.4f},{seltime:.4f},{entime:.4f}'
                  .format(ep=rnd, itr=-1, loss=test_loss, trainloss=train_loss,
                          filler=-1, val=test_acc, other=train_loss1, updtime=update_time, comptime=comp_time,
                          seltime=sel_time, entime=update_time+comp_time+sel_time), file=f)

def evaluate_client(model, criterion, partition):

    '''
    Evaluate each client on their local train dataset against the current global model

    Evaluating each client's local loss values for the current global model for client selection
    :param model: current global model
    :param criterion: loss function
    :param partition: dataset dict for clients
    :return: cli_loss = list of local loss values, cli_comptime = list of computation time
    '''

    cli_comptime, cli_loss = [], []
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
                                                   pin_memory=True)

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
            cli_comptime.append(comptime_end-comptime_start)
            cli_loss.append(final_loss)

    return cli_loss, cli_comptime

def evaluate(model, test_loader, criterion):
    """
    Evaluate test accuracy
    Evaluate model on full test dataset
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # Get test accuracy for the current model
    with torch.no_grad():
        # print("****", type(test_loader))
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
            logging.debug('epoch {} itr {}, '
                         'client_idx {}, loss value {:.4f}, train accuracy {:.3f}'
                         .format(epoch, batch_idx, client_idx, los, acc))

            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},'
                      '{loss:.4f},-1,-1,'
                      '{top1:.3f},-1,-1,-1,-1,-1,-1'
                      .format(ep=epoch, itr=batch_idx,
                              loss=los, top1=acc), file=f)

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},'
              '{loss:.4f},-1,-1,'
              '{top1:.3f},-1,-1,-1,-1,-1,-1'
              .format(ep=epoch, itr=batch_idx,
                      loss=los, top1=acc), file=f)

    return los

# def update_learning_rate(optimizer, epoch, target_lr):
#     """
#     Decay learning rate
#     ** note: target_lr is the reference learning rate from which to scale down
#     """
#     if epoch == 149:
#         lr = target_lr/2
#         logging.info('Updating learning rate to {}'.format(lr))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

#     if epoch == 299:
#         lr = target_lr/4
#         logging.info('Updating learning rate to {}'.format(lr))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

def init_processes(rank, size, world_size, fn):
    """ Initialize the distributed environment. """
    
    print('rank {} size {}'.format(rank, size))
    dist.init_process_group(backend=args.backend, 
                            init_method=args.initmethod, 
                            rank=rank, 
                            world_size=world_size)
    fn(rank, size)

if __name__ == "__main__":
    rank = args.rank
    # clients_per_round = args.clients_per_round
    # world_size = args.world_size

    # mp.spawn(init_processes, args=(size, world_size, run), nprocs=world_size, join=True)
    # # init_processes(rank, size, run)
    run(rank, args)
