import os
import json
from tqdm import tqdm
import numpy as np

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from FedAvg import *

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

## hyperparameters
train_data_dir = './synthetic_data/'
test_data_dir = './synthetic_data/'
lr = 0.05  # learning rate, \eta
bs = 50  # batch size, b
le = 30  # local epoch, E
total_rnd = 800  # total communication rounds, T/E
sample_ratio = 3  # clients per round, m
K = 30  # number of clients, K

## experiment configurations
# key=experiment_id, value=(algo, powd, color, linestyle)
client_selection_type = {
    'rand': ('rand', 1, 'k', '-'),
    'powd2': ('pow-d', sample_ratio*2, c_t(3), '-.'),
    'powd5': ('pow-d', sample_ratio*10, c_t(0), '--'),
    'adapow30': ('adapow-d', K, c_t(1), (0, (5, 10)))
}

# make logs directory if not exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

## run experiments
for key in client_selection_type.keys():
    # set seed for reproducibility
    np.random.seed(12345)

    # fetch configuration
    algo, powd, color, lstyle = client_selection_type[key]

    # run federated learning experiment for given configuration
    opt = FedAvg(lr, bs, le, algo, powd, train_data_dir, test_data_dir, sample_ratio)
    errors, local_errors = list(), list()
    for rnd in tqdm(range(total_rnd), desc=key): 
        # reduce learning rate by half after 300 and 600 rounds
        if rnd == 300 or rnd == 600:
            opt.lr /= 2

        # reduce powd from K to m after half rounds for 'adapow-d'
        if algo == 'adapow-d' and rnd == total_rnd//2:
            opt.powd = sample_ratio

        # execute one communication round
        weights, _ = opt.local_update(local_errors)

        # update global parameter by aggregating weights
        opt.aggregate(weights)

        # evaluate global and local losses
        error, local_errors = opt.evaluate()
        errors.append(error)

    # # save errors to json file
    # with open(f'./logs/m={sample_ratio}_algo={key}_errors.json', 'w') as f:
    #     json.dump(errors, f)
    
    # # load errors from json file
    # with open(f'./logs/m={sample_ratio}_algo={key}_errors.json') as f:
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
plt.title('K=30, m={}'.format(sample_ratio))
# plt.show()
plt.savefig(f'synthetic_m={sample_ratio}.pdf', bbox_inches='tight')