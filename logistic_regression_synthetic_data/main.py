import numpy as np
from tqdm import tqdm
import json
import os

from optimizer import *
from FedAvg import *

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib.pylab as pylab

# hyperparameters
train_data_dir = './synthetic_data/'
test_data_dir = './synthetic_data/'
lr = 0.05  # learning rate, \eta
bs = 50  # batch size, b
le = 30  # local epoch, E
total_rnd = 100  # total communication rounds, T/E
sample_ratio = 1  # clients per round, m

# etamu = 0
# color maps reference: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
c_t = cm.get_cmap('tab10')

# key=key, value=(sel_type, powd, color, linestyle)
client_selection_type = {
    'rand': ('rand', 1, 'k', '-'),
    'powd2': ('pow-d', sample_ratio*2, c_t(3), '-.'),
    'powd5': ('pow-d', sample_ratio*10, c_t(0), '--'),
    'adapow30': ('adapow-d', 30, c_t(1), (0, (5, 10)))
}

# run_keys = ['rand', 'powd2', 'powd5', 'adapow30']
# run_keys = ['adapow30']

# plot settings
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

if not os.path.exists('./logs'):
    os.makedirs('./logs')

# run experiments
for key in client_selection_type.keys():
    print("running {}".format(key))
    np.random.seed(12345)
    # sel_freq = np.zeros(30)  # number of times each client is selected
    algo, powd, color, lstyle = client_selection_type[key]

    # run federated learning experiment for given configuration
    opt = FedAvg(lr, bs, le, algo, powd, train_data_dir, test_data_dir, sample_ratio)
    errors, local_errors = list(), list()
    for rnd in tqdm(range(total_rnd)): 
        # reduce learning rate by half after 300 and 600 rounds
        if rnd == 300 or rnd == 600:
            opt.lr /= 2

        # reduce powd from K to m after half rounds
        if algo == 'adapow-d' and rnd == total_rnd//2:
            opt.powd = sample_ratio

        delta, _ = opt.local_update(local_errors)
        # for i in workers:
        #     sel_freq[i]+=1

        # update central parameter by aggregating deltas
        opt.aggregate(delta)

        # evaluate global and local losses
        error, local_errors = opt.evaluate()
        errors.append(error)

    # save errors to json file
    with open(f'./logs/m={sample_ratio}_algo={key}_errors.json', 'w') as f:
        json.dump(errors, f)
    
    # # load from json file
    # with open(f'./logs/m={sample_ratio}_algo={key}_errors.json') as f:
    #     errors = json.load(f)

    # all_sel_freq.append(sel_freq)
    
    # plotting pursposes
    if algo =='rand' or algo =='adapow-d':
        p_label = algo
    else:
        p_label = algo+', d={}'.format(powd)

    plt.plot(errors, lw=lw, color=color, ls = lstyle, label=p_label)
    # plt.plot(errors, lw=lw, color=color, label=p_label)

print("plotting...")
plt.ylabel('Global loss')
plt.xlabel('Communication round')
legend_properties = {'weight':'bold'}
plt.xticks()
plt.yticks()
plt.legend(loc=1)
plt.grid()
plt.title('K=30, m={}'.format(sample_ratio))
# plt.show()

plt.savefig(f'synthetic_m={sample_ratio}.pdf', bbox_inches='tight')