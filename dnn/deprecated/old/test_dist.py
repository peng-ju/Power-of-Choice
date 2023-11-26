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

# from distoptim import fedavg
# import util_v4 as util
# import models
# from params import args_parser


# logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
# logging.debug('This message should appear on the console')

# args = args_parser()

def run(rank, size, device=None):
    print('in run: rank={}, size={}, device: {}'.format(rank, size, device))

    assert size == dist.get_world_size(), "size and dist.get_world_size() don't match"

    # mytensor = [torch.tensor(0, dtype=torch.float32) for _ in range(dist.get_world_size())]
    # mytensor = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    # print('mytensor:', mytensor)
    # print('dtype:', mytensor.dtype)
    # print('dtype [0]:', mytensor[0].dtype)

    # print(f'rank: {rank}, barrier 1')
    # dist.barrier()
    # select client for each round, in total m ranks
    send = torch.zeros(size, dtype=torch.float32).to("cpu")
    if rank == 0:
        # print('master process selecting clients')
        # idxs_users = np.random.choice(100, size=size)
        # print('idxs_users:', idxs_users)
        # send = torch.tensor(idxs_users, dtype=torch.int)
        send = torch.ones(size, dtype=torch.float32).to("cpu")
        send = send + 3.2
        print('master process send:', send)
    print(f'rank: {rank}, initial send: {send}')
    # print(f'rank: {rank}, barrier 2')
    # dist.barrier()

    if rank == 3:
        t=20
        print(f'rank {rank} sleeping for {t} seconds')
        time.sleep(t)
    dist.broadcast(tensor=send, src=0)
    print('[broadcast] rank:', rank, 'send:', send, 'send.dtype:', send.dtype, 'send.device:', send.device)

    send = torch.tensor([7 for _ in range(size)], dtype=torch.int).to(device)
    print(f'rank: {rank}, updated send: {send}, dtype: {send.dtype}')
    # del send

    # if rank == 3:
    #     t=10
    #     print(f'rank {rank} sleeping for {t} seconds')
    #     time.sleep(t)
    # return f'rank: {rank}, done'
    

def init_processes(rank, size, device, num_devices, fn):
    """ Initialize the distributed environment. """

    dist.init_process_group(backend='gloo', 
                            init_method='tcp://localhost:29500', 
                            rank=rank, 
                            world_size=size)
    device = "cuda"  # torch.device("cuda:{}".format(rank%num_devices)) if torch.cuda.is_available() else torch.device("cpu")
    fn(rank, size, device)
    # print('fn is run for rank:', rank)

if __name__ == "__main__":
    size = 8
    print('about to spawn')
    # print(args)

    start_time = time.time()
    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = None
    num_devices = torch.cuda.device_count()
    print('main: number of gpus:', num_devices)
    print('main: device:', device)
    mp.spawn(init_processes, args=(size, device, num_devices, run), nprocs=size, join=True)
    print('time taken:', time.time() - start_time)
    print('done')

