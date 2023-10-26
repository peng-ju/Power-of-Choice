import argparse

def args_parser():
    """
    parser for command-line options, arguments and sub-commands
    """
    parser = argparse.ArgumentParser(description='quadratic_sim')
    parser.add_argument('--name', '-n',
                        default="default",
                        type=str,
                        help='experiment name, used for saving results')
    parser.add_argument('--num_users',
                        default=30,
                        type=int,
                        help='number of entire clients')
    parser.add_argument('--frac',
                        default=0.1,
                        type=float,
                        help='selection fraction')
    parser.add_argument('--powd',
                        default=30,
                        type=int,
                        help='number of selected subset workers per round')
    parser.add_argument('--alpha',
                        default=3,
                        type=float,
                        help='control the non-iidness of dataset')
    parser.add_argument('--seed',
                        default=2,
                        type=int,
                        help="for reproducibility")
    parser.add_argument('--high',
                        default=9,
                        type=float,
                        help='for H matrix generation')
    parser.add_argument('--lr',
                        default=0.00002,
                        type=float,
                        help='learning rate')
    parser.add_argument('--le',
                        default=2,
                        type=int,
                        help='local updates for each client')
    parser.add_argument('--epochs',
                        default=15000,
                        type=int,
                        help='total communication rounds')
    parser.add_argument('--seltype',
                        default='rand',
                        type=str,
                        help='type of client selection')
    parser.add_argument('--dim',
                        default=5,
                        type=int,
                        help='dimension of data')
    parser.add_argument('--eq',
                        default=0,
                        type=int,
                        help='dataset is equalornot')

    args = parser.parse_args()

    return args
