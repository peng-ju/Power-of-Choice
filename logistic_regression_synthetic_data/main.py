import os
import sys
import json
from tqdm import tqdm
import numpy as np

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mlflow
from setup_mlflow import get_experiment_id, MLFLOW_TRACKING_URI

from FedAvg import FedAvg

## hyperparameters
train_data_dir = '../data/synthetic_data/'
test_data_dir = '../data/synthetic_data/'
lr = 0.05  # learning rate, \eta
bs = 50  # batch size, b
le = 30  # local epoch, E
total_rnd = 800  # total communication rounds, T/E
clients_per_round = int(sys.argv[1])  # active clients per round, m
K = 30  # number of clients, K
SEED = 12345
log_remote = False

## create logs directory if not exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

## plot settings
# color maps reference: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
# line styles reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
c_t = cm.get_cmap('tab10')
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

## experiment configurations
# key=experiment_id, value=(algo, powd, color, linestyle)
client_selection_type = {
    'rand': ('rand', 1, 'k', '-'),
    'powd2': ('pow-d', clients_per_round*2, c_t(3), '-.'),
    'powd5': ('pow-d', clients_per_round*10, c_t(0), '--'),
    'adapow30': ('adapow-d', K, c_t(1), (0, (5, 10)))
}

## run experiments
if log_remote: mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
with mlflow.start_run(experiment_id=get_experiment_id('test1')):
    # log hyperparameters
    mlflow.log_params({
        'lr': lr,
        'bs': bs,
        'le': le,
        'total_rnd': total_rnd,
        'clients_per_round': clients_per_round,
        'K': K,
        'SEED': SEED
    })

    for key in client_selection_type.keys():
        # set seed for reproducibility
        np.random.seed(SEED)

        # fetch configuration
        algo, powd, color, lstyle = client_selection_type[key]

        # run federated learning experiment for given configuration
        server = FedAvg(lr, bs, le, algo, powd, train_data_dir, test_data_dir, clients_per_round)
        errors, local_losses = [], []
        for rnd in tqdm(range(total_rnd), desc=key): 
            # reduce learning rate by half after 300 and 600 rounds
            if rnd == 300 or rnd == 600:
                for param_group in server.optimizer.param_groups:
                    param_group["lr"] /= 2

            # reduce powd from K to m after half rounds (only for 'adapow-d')
            if algo == 'adapow-d' and rnd == total_rnd//2:
                server.powd = clients_per_round

            # find the set of active clients
            active_clients = server.select_client(local_losses)

            # train active clients locally
            weights, _ = server.local_update(active_clients)

            # update global parameter by aggregating weights
            server.aggregate(weights)

            # evaluate global and local losses
            global_loss, local_losses = server.evaluate()
            mlflow.log_metric(f'global_loss_{algo}', global_loss, step=rnd)
            errors.append(global_loss)

        # save errors to json file
        with open(f'./logs/m={clients_per_round}_algo={key}_errors.json', 'w') as f:
            json.dump(errors, f)
        
        # # load errors from json file
        # with open(f'./logs/m={clients_per_round}_algo={key}_errors.json') as f:
        #     errors = json.load(f)
        
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

    mlflow.log_artifact(f'synthetic_m={clients_per_round}.pdf')
    mlflow.log_artifact(f'./logs/')