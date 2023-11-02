import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='FMNIST baseline')
    parser.add_argument('--name', '-n',
                        default="default",
                        type=str,
                        help='experiment name, used for saving results')
    parser.add_argument('--backend',
                        default="gloo",  # ncll, gloo
                        type=str,
                        help='backend name')
    parser.add_argument('--model',
                        default="MLP",
                        type=str,
                        help='neural network model')
    parser.add_argument('--alpha',
                        default=2,  # 0.3, 2
                        type=float,
                        help='control the non-iidness of dataset')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='number of classes')
    parser.add_argument('--gmf',
                        default=0,
                        type=float,
                        help='global (server) momentum factor')
    parser.add_argument('--lr',
                        default=0.005,
                        type=float,
                        help='client learning rate')
    parser.add_argument('--momentum',
                        default=0.0,
                        type=float,
                        help='local (client) momentum factor')
    parser.add_argument('--bs',
                        default=64,
                        type=int,
                        help='batch size on each worker/client')
    parser.add_argument('--rounds',
                        default=100,
                        type=int,
                        help='total communication rounds')
    parser.add_argument('--localE',
                        default=30,
                        type=int,
                        help='number of local epochs')
    parser.add_argument('--decay',
                        default=1,
                        type=bool,
                        help='1: decay LR, 0: no decay')
    parser.add_argument('--print_freq',
                        default=5,
                        type=int,
                        help='print info frequency')
    parser.add_argument('--size',
                        default=3,
                        type=int,
                        help='number of local workers')
    parser.add_argument('--powd',
                        default=6,
                        type=int,
                        help='number of selected subset workers per round ($d$)')
    parser.add_argument('--fracC',
                        default=0.03,
                        type=float,
                        help='fraction of selected workers per round')
    parser.add_argument('--seltype',
                        default='pow-d',  # rand, pow-d
                        type=str,
                        help='type of client selection ($\pi$)')
    parser.add_argument('--ensize',
                        default=100,
                        type=int,
                        help='number of all workers')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='the rank of worker')
    parser.add_argument('--rnd_ratio',
                        default=0.1,
                        type=float,
                        help='hyperparameter for afl')
    parser.add_argument('--delete_ratio',
                        default=0.75,
                        type=float,
                        help='hyperparameter for afl')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed')
    parser.add_argument('--save', '-s',
                        action='store_true',
                        help='whether save the training results')
    parser.add_argument('--p', '-p',
                        action='store_true',
                        help='whether the dataset is partitioned or not')
    parser.add_argument('--NIID',
                        action='store_false',  # TODO
                        help='whether the dataset is non-iid or not')
    parser.add_argument('--commE',
                        action='store_true',
                        help='activation of $cpow-d$')
    parser.add_argument('--constantE',
                        action='store_false',  # TODO
                        help='whether all the local workers have an identical \
                        number of local epochs or not')
    parser.add_argument('--optimizer',
                        default='fedavg',
                        type=str,
                        help='optimizer name')
    parser.add_argument('--initmethod',
                        default='tcp://localhost:29500',
                        type=str,
                        help='init method')
    parser.add_argument('--mu',
                        default=0,
                        type=float,
                        help='mu parameter in fedprox')
    parser.add_argument('--dataset',
                        default='fmnist',
                        type=str,
                        help='type of dataset')

    args = parser.parse_args()

    return args