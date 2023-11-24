import os
import numpy as np
import random

import time
import pathlib
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from distoptim import fedavg
import util_text as util
import models
from params import args_parser

# define device
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logging.debug('This message should appear on the console')

args = args_parser()

def run(rank, size):
    print("run start \n")
    # initiate experiments folder
    save_path = './logs/'
    fold = 'lr{:.4f}_bs{}_cp{}_a{:.2f}_e{}_r0_n{}_f{:.2f}/'.format(args.lr, 
                                                                   args.bs, 
                                                                   args.localE, 
                                                                   args.alpha, 
                                                                   args.seed,
                                                                   args.ensize, 
                                                                   args.fracC)
    if args.commE:
        fold = 'com_'+fold
    folder_name = save_path+args.name+'/'+fold
    file_name = '{}_rr{:.2f}_dr{:.2f}_lr{:.3f}_bs{:d}_cp{:d}_a{:.2f}_e{}_r{}_n{}_f{:.2f}_p{}.csv'.format(args.seltype,
                                                                                                         args.rnd_ratio, 
                                                                                                         args.delete_ratio, 
                                                                                                         args.lr, 
                                                                                                         args.bs, 
                                                                                                         args.localE,
                                                                                                         args.alpha, 
                                                                                                         args.seed, 
                                                                                                         rank, 
                                                                                                         args.ensize, 
                                                                                                         args.fracC, 
                                                                                                         args.powd)
                                                    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # initiate log files
    saveFileName = folder_name + file_name
    args.out_fname = saveFileName
    with open(args.out_fname, 'w+') as f:
        print('Epoch,itr,loss,trainloss,avg:Loss,Prec@1,avg:Prec@1,val,trainval,updtime,comptime,seltime,entime,testacc,testloss', file=f)

    # seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    partition, train_loader, test_loader, dataratios, traindata = util.partition_dataset(size, args, 0)
    print("\n dataratios: ", dataratios)

    # tracking client loss values, frequency for each client
    # args.ensize -- number of clients
    client_freq, client_loss_proxy = np.zeros(args.ensize), np.zeros(args.ensize)

    # initialization for client selection
    cli_loss, cli_freq, cli_val = np.zeros(args.ensize)+1, np.zeros(args.ensize), np.zeros(args.ensize)

    # select client for the 1st round
    replace_param = False
    if args.seltype =='rand':
        replace_param = True

    # user id being selected
    sel_idx = np.random.choice(args.ensize, size=args.size, replace=replace_param)

    # define multilayer perceptron neural network model for sentiment analysis
    model = models.MLP_text(input_size=200, dim_hidden1=128, dim_hidden2=86, dim_hidden3=30, dim_out=1).to(device)
    
    # allocate buffer for global and aggregate parameters
    # ref: https://discuss.pytorch.org/t/how-to-assign-an-arbitrary-tensor-to-models-parameter/44082/3
    global_parameters = []
    aggregate_parameters = []
    with torch.no_grad():
        for param in model.parameters():
            global_parameters.append(param.detach().clone())
            aggregate_parameters.append(torch.zeros_like(param)) 
    
    # criterion
    # criterion = nn.NLLLoss().to(device)  # for multi-class classifier.
    criterion =nn.BCELoss().to(device)  # for binary classifier

    # select optimizer according to algorithm
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=args.momentum, 
                                nesterov=False,
                                weight_decay=1e-4)

    test_loss_rnd = []
    test_accu_rnd = []
    train_loss_rnd = []
    train_accu_rnd = []


    # start communication rounds
    for rnd in range(args.rounds):
        round_start = time.time()

        # (optional) decay learning rate according to round index
        # if args.decay == True:
        #     # update_learning_rate(optimizer, rnd, args.lr)
        #     if rnd == 160:
        #         lr = args.lr/2
        #         logging.info("Updating learning rate to {}".format(lr))
        #         for param_group in optimizer.param_groups:
        #             param_group["lr"] = lr

        #     if rnd == 299:
        #         lr = args.lr/4
        #         logging.info("Updating learning rate to {}".format(lr))
        #         for param_group in optimizer.param_groups:
        #             param_group["lr"] = lr

        # zero aggregate parameters for accumulation of local parameters
        with torch.no_grad():
            for param in aggregate_parameters:
                param.zero_()

        # for each client `i`
        for i in sel_idx:
            # send global parameters to client `i`
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_parameters):
                    param.copy_(global_param)
            
            # run E steps of SGD on client `i`
            loss_final = 0
            comm_update_start = time.time()
            for t in range(args.localE):
                singlebatch_loader = util.partitiondata_loader(partition, i, args.bs, traindata)
                loss, model = train_text(rank, model, criterion, optimizer, singlebatch_loader, t)
                loss_final += loss/args.localE #average over localE iterations
            comm_update_end = time.time()
            update_time = comm_update_end - comm_update_start

            # send local parameters from client `i` to server for aggregation
            with torch.no_grad():
                weight = 1/args.size
                for aggregate_param, param in zip(aggregate_parameters, model.parameters()):
                    aggregate_param.add_(param, alpha=weight)
            
            # update client frequency and loss values
            client_freq[i] += 1
            client_loss_proxy[i] = loss_final

        not_visited = np.where(client_freq == 0)[0]
        for j in not_visited:
            if args.seltype == "afl":
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
        cli_loss, cli_comptime = evaluate_client(model, criterion, partition, traindata)
        train_loss = sum([cli_loss[i]*dataratios[i] for i in range(args.ensize)])
        # train_loss1 = sum(cli_loss)/args.ensize

        # select clients for the next round
        sel_time, comp_time = 0, 0
        sel_time_start = time.time()
        """
        noteL cli_val, rnd are not useful?
        """
        sel_idx, rnd_idx = util.sel_client(dataratios, cli_loss, cli_val, args, rnd)
        # print(f"len rnd_idx {len(rnd_idx)} idxs_users {len(idxs_users)}")
        sel_time_end = time.time()
        sel_time = sel_time_end - sel_time_start

        if args.seltype == "pow-d" or args.seltype == "pow-dint":
            comp_time = max([cli_comptime[int(i)] for i in rnd_idx])

        # record metrics
        round_end = time.time()
        round_duration = round(round_end - round_start, 1)
        logging.info(f"[{round_duration} s] Round {rnd} rank {rank} test accuracy {test_acc:.3f} test loss {test_loss:.3f} train loss {train_loss:.3f}")

        test_loss_rnd.append(test_loss)
        test_accu_rnd.append(test_acc)
        test_loss_rnd.append(train_loss)
        # test_accu_rnd.append(train_acc)
        
        # itr = -1 for overal result
        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},{loss:.4f},{trainloss:.4f},{filler},'
                  '{filler},{filler},'
                  '{val:.4f},{other:.4f},{updtime:.4f},{comptime:.4f},{seltime:.4f},{entime:.4f}, {testacc:.4f}, {testloss:.4f}'
                  .format(ep=rnd, itr=-1, loss=test_loss, trainloss=train_loss,
                          filler=-1, val=test_acc, other=train_loss, updtime=update_time, comptime=comp_time,
                          seltime=sel_time, entime=update_time+comp_time+sel_time, testacc=test_acc, testloss=test_loss), file=f)


def evaluate_client(model, criterion, partition, traindata):

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
        partitioned = partition[i]

        # cpow-d
        if args.commE:
            seldata_idx = random.sample(range(len(partitioned)), k=int(min(args.bs, len(partitioned))))
        else:
            seldata_idx = partitioned

        other = torch.utils.data.Subset(traindata, indices=seldata_idx)
        train_loader = torch.utils.data.DataLoader(other, batch_size=args.bs, shuffle=False,
                                                    pin_memory=True)

        # Compute local loss values or proxies for the clients
        tmp, total = 0,0
        with torch.no_grad():
            comptime_start = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device,non_blocking=True)
                target = target.to(device,non_blocking=True)
                vec_target = vector_encoding(target)

                vec_target = vec_target.to(device,non_blocking=True)

                outputs = model(data)
                outputs.to(device)
                loss = criterion(outputs, vec_target)
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

            data = data.to(device,non_blocking=True)
            target = target.to(device,non_blocking=True)
            vec_target = vector_encoding(target)

            
            vec_target = vec_target.to(device,non_blocking=True)

            # Inference
            outputs = model(data)
            outputs.to(device)
            batch_loss = criterion(outputs, vec_target)
            loss += batch_loss.item()

            # Prediction
            # _, pred_labels = torch.max(outputs, 1)
            pred_labels = get_label(outputs)
            correct += torch.sum(torch.eq(pred_labels, vec_target)).item()/len(pred_labels)
            total += 1

        acc = (correct / total) * 100
        los = loss/total

    return acc, los


def train_text(rank, model, criterion, optimizer, loader, epoch):
    """
    train model on the sampled mini-batch for $\tau$ epochs
    """

    model.train()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        data = data.to(device,non_blocking = True)
        target = target.to(device,non_blocking = True)
        # encode target
        vec_target = vector_encoding(target)

        vec_target = vec_target.to(device,non_blocking = True)
        
        outputs = model(data)
        outputs.to(device)
        # print("\n outputs: ", outputs)
        # print("\n vec_target: ", vec_target)
        batch_loss = criterion(outputs, vec_target)

        # backward pass
        batch_loss.backward()

        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # write log files
        loss += batch_loss.item()

        # Prediction
        # _, pred_labels = torch.max(outputs, 1)
        pred_labels = get_label(outputs)

        correct += torch.sum(torch.eq(pred_labels, vec_target)).item()/len(pred_labels)
        total += 1

        acc = (correct / total)*100
        los = loss / total

        # if batch_idx % args.print_freq == 0 and args.save:
        #     logging.debug('epoch {} itr {}, '
        #                  'rank {}, loss value {:.4f}, train accuracy {:.3f}'
        #                  .format(epoch, batch_idx, rank, los, acc))

        #     with open(args.out_fname, '+a') as f:
        #         print('{ep},{itr},'
        #               '{loss:.4f},-1,-1,'
        #               '{top1:.3f},-1,-1,-1,-1,-1,-1'
        #               .format(ep=epoch, itr=batch_idx,
        #                       loss=los, top1=acc), file=f)

        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},'
                '{loss:.4f},-1,-1,'
                '{top1:.3f},-1,-1,-1,-1,-1,-1'
                .format(ep=epoch, 
                        itr=batch_idx,
                        loss=los, 
                        top1=acc), file=f)

    
    return los, model

def get_label(x):
    """
    x : probability of been positive
    return: predicted label
    """
    res = torch.zeros_like(x)
    res[x>0.5] = 1
    
    return res

def vector_encoding(target):
    """
    This is a binary classification

    # 0 for positive, 1 for negative
    """
    vector = torch.Tensor([1-i.item() for i in target])
    # vector = torch.Tensor([i.item() for i in target])
    return vector


if __name__ == "__main__":
    rank = args.rank
    size = args.size

    run(rank, size)

