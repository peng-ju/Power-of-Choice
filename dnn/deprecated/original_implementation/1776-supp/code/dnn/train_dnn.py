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

args = args_parser()

def run(rank, size):
    print('in run: rank={}, size={}'.format(rank, size))
    # initiate experiments folder
    save_path = './logs/'
    fold = 'lr{:.4f}_bs{}_cp{}_a{:.2f}_e{}_r0_n{}_f{:.2f}/'.format(args.lr, args.bs, args.localE, args.alpha, args.seed,
                                                                   args.ensize, args.fracC)
    if args.commE:
        fold = 'com_'+fold
    folder_name = save_path+args.name+'/'+fold
    file_name = '{}_rr{:.2f}_dr{:.2f}_lr{:.3f}_bs{:d}_cp{:d}_a{:.2f}_e{}_r{}_n{}_f{:.2f}_p{}.csv'.format(args.seltype,
                                                     args.rnd_ratio, args.delete_ratio, args.lr, args.bs, args.localE,
                                                    args.alpha, args.seed, rank, args.ensize, args.fracC, args.powd)
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # initiate log files
    saveFileName = folder_name+file_name
    args.out_fname = saveFileName
    with open(args.out_fname, 'w+') as f:
        print('BEGIN-TRAINING\n' 'World-Size,{ws}\n' 'Batch-Size,{bs}\n' 'Epoch,itr,'
            'loss,trainloss,avg:Loss,Prec@1,avg:Prec@1,val,trainval,updtime,comptime,seltime,entime'.format(
            ws=args.size, bs=args.bs), file=f)

    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    partition, train_loader, test_loader, dataratios, datstat, endat = util.partition_dataset(size, args, 0)

    # initialization for client selection
    cli_loss, cli_freq, cli_val = np.zeros(args.ensize)+1, np.zeros(args.ensize), np.zeros(args.ensize)

    tmp_cli = [torch.tensor(0, dtype=torch.float32) for _ in range(dist.get_world_size())]
    tmp_clifreq = [torch.tensor(0, dtype=torch.int32) for _ in range(dist.get_world_size())]

    dist.barrier()
    # select client for each round, in total m ranks
    send = torch.zeros(args.size, dtype=torch.int32)
    if rank == 0:
        replace_param = False
        if args.seltype =='rand':
            replace_param = True

        idxs_users = np.random.choice(args.ensize, size=args.size, replace=replace_param)
        send = [torch.tensor(int(ii), dtype=torch.int32) for ii in idxs_users]
    dist.barrier()

    for i in range(args.size):
        dist.broadcast(tensor=send[i], src=0)
    dist.barrier()
    sel_idx = int(send[rank])
    print(f'rank: {rank}, send: {send}, sel_idx: {sel_idx}')

    # define neural nets model, criterion, and optimizer
    len_in = 1
    for x in args.img_size:
        len_in *= x

    if args.model == 'MLP':
        model = models.MLP_FMNIST(dim_in=len_in, dim_hidden1=64, dim_hidden2 = 30, dim_out=args.num_classes)

    elif args.model == 'CNN':
        model = models.CNNCifar(args)  # vgg

    criterion = nn.NLLLoss()

    # select optimizer according to algorithm
    algorithms = {'fedavg': fedavg}

    selected_opt = algorithms[args.optimizer]
    optimizer = selected_opt(model.parameters(),
                      lr=args.lr,
                      gmf=args.gmf, # set to 0
                      mu = args.mu, # set to 0
                      ratio=-1,  # dataratios[rank],
                      momentum=args.momentum, # set to 0
                      nesterov = False,
                      weight_decay=1e-4)

    print('about to start training')
    for rnd in range(args.rounds):
        print('round:', rnd)

        # Initialize hyperparameters
        local_epochs = args.localE
        weight = 1/args.size

        # Decay learning rate according to round index (optional)
        if args.decay == True:
            update_learning_rate(optimizer, rnd, args.lr)

        # Clients locally train for several local epochs
        loss_final = 0
        dist.barrier()
        comm_update_start = time.time()
        for t in range(local_epochs):
            # print(f'rank: {rank}, rnd: {rnd}, t: {t}, sel_idx: {sel_idx}')
            singlebatch_loader = util.partitiondata_loader(partition, sel_idx, args.bs)
            loss = train(model, criterion, optimizer, singlebatch_loader, t, rank)
            loss_final += loss/local_epochs
        dist.barrier()
        comm_update_end = time.time()
        update_time = comm_update_end - comm_update_start

        # Getting value function for client selection (required only for 'rpow-d', 'afl')
        dist.barrier()      # TODO: implement with multi-arm bandit
        dist.all_gather(tmp_cli, torch.tensor(loss_final))
        dist.all_gather(tmp_clifreq, torch.tensor(int(sel_idx), dtype=torch.int32))
        dist.barrier()
        for i, i_val in enumerate(tmp_clifreq):
            cli_freq[i_val.item()]+= 1         # Cli freq is the entire clients that are selected for all rounds
            cli_val[i_val.item()] = tmp_cli[i].item()
        not_visited = np.where(cli_freq == 0)[0]

        for ii in not_visited:
            if args.seltype == 'afl':
                cli_val[ii] = -np.inf
            else:
                cli_val[ii] = np.inf

        # synchronize parameters
        dist.barrier()
        optimizer.average(weight=weight)
        dist.barrier()

        # evaluate test accuracy
        test_acc, test_loss = evaluate(model, test_loader, criterion)

        # evaluate loss values and sync selected frequency
        cli_loss, cli_comptime = evaluate_client(model, criterion, partition)
        train_loss = sum([cli_loss[i]*dataratios[i] for i in range(args.ensize)])
        train_loss1 = sum(cli_loss)/args.ensize

        dist.barrier()
        # Select client for each round, in total m ranks
        send = torch.zeros(args.size, dtype=torch.int32)
        comp_time, sel_time = 0, 0

        if rank == 0:
            sel_time_start = time.time()
            idxs_users, rnd_idx = util.sel_client(dataratios, cli_loss, cli_val, args, rnd)
            sel_time_end = time.time()
            sel_time = sel_time_end - sel_time_start

            if args.seltype == 'pow-d' or args.seltype == 'pow-dint':
                comp_time = max([cli_comptime[int(i)] for i in rnd_idx])

            send = [torch.tensor(int(ii), dtype=torch.int32) for ii in idxs_users]
        dist.barrier()
        for i in range(args.size):
            dist.broadcast(tensor=send[i], src=0)
        dist.barrier()
        sel_idx = int(send[rank])
        print(f'next round rank: {rank}, send: {send}, sel_idx: {sel_idx}')

        # record metrics
        logging.info("Round {} rank {} test accuracy {:.3f} test loss {:.3f}".format(rnd, rank, test_acc, test_loss))
        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},{loss:.4f},{trainloss:.4f},{filler},'
                  '{filler},{filler},'
                  '{val:.4f},{other:.4f},{updtime:.4f},{comptime:.4f},{seltime:.4f},{entime:.4f}'
                  .format(ep=rnd, itr=-1, loss=test_loss, trainloss=train_loss,
                          filler=-1, val=test_acc, other=train_loss1, updtime=update_time, comptime=comp_time,
                          seltime=sel_time, entime=update_time+comp_time+sel_time), file=f)

def evaluate_client(model, criterion, partition):

    '''
    Evaluating each client's local loss values for the current global model for client selection
    :param model: current global model
    :param criterion: loss function
    :param partition: dataset dict for clients
    :return: cli_loss = list of local loss values, cli_comptime = list of computation time
    '''

    cli_comptime, cli_loss = [], []
    model.eval()

    # Get data from client to evaluate local loss on
    for i in range(args.ensize):
        partitioned = partition.use(i)

        # cpow-d
        if args.commE:
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
                # data = data.cuda(non_blocking=True)
                # target = target.cuda(non_blocking=True)
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
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # Get test accuracy for the current model
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # data = data.cuda(non_blocking = True)
            # target = target.cuda(non_blocking = True)

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


def train(model, criterion, optimizer, loader, epoch, rank):
    """
    train model on the sampled mini-batch for $\tau$ epochs
    """

    model.train()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        # data = data.cuda(non_blocking = True)
        # target = target.cuda(non_blocking = True)

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
                         'rank {}, loss value {:.4f}, train accuracy {:.3f}'
                         .format(epoch, batch_idx, rank, los, acc))

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

def update_learning_rate(optimizer, epoch, target_lr):
    """
    Decay learning rate
    ** note: target_lr is the reference learning rate from which to scale down
    """
    if epoch == 149:
        lr = target_lr/2
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == 299:
        lr = target_lr/4
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """

    dist.init_process_group(backend=args.backend, 
                            init_method=args.initmethod, 
                            rank=rank, 
                            world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    rank = args.rank
    size = args.size

    # init_processes(rank, size, run)
    print(args)
    mp.spawn(init_processes, args=(size, run), nprocs=size, join=True)
    print('done')

