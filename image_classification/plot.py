import os
import re
import ast
import argparse

from glob import glob
import pandas as pd

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
c_t = cm.get_cmap('tab10')


def make_plot(log_filenames, metric='train_loss', niters=None, save=True, save_path=None):
    ## plot settings
    # color maps reference: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    # line styles reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    fontsize = 20
    params = {'legend.fontsize': fontsize,
            'axes.labelsize': fontsize,
            'axes.titlesize':fontsize,
            'xtick.labelsize':fontsize,
            'ytick.labelsize':fontsize}
    plt.rcParams.update(params)
    lw = 2
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.labelweight'] = 'bold'
    # plt.figure(figsize=(16,14.5))
    plt.figure()
    plt.subplots_adjust(right=1.1, top=0.9)
    rcParams['axes.titlepad'] = 14

    for log_filename in log_filenames:
        # load metric data from json file and fetch configuration
        start_idx = 0
        plot_label = None
        with open(log_filename, 'r') as f:
            for line in f:
                line = line.strip()  # to remove '\n' at the end of line
                if line.startswith('rank,round,epoch,test_loss'):
                    break
                elif line.startswith('algo'):
                    algo = line.split(',')[1]
                elif line.startswith('powd'):
                    powd = int(line.split(',')[1])
                elif line.startswith('clients_per_round'):
                    clients_per_round = int(line.split(',')[1])
                elif line.startswith('name'):
                    name = line.split(',')[1]
                elif line.startswith('plot_label'):
                    plot_label = line.split(',')[1]
                elif line.startswith('plot_linecolor'):
                    color = line[len('plot_linecolor,'):]
                    if color.startswith('(') or color.startswith('c_t('):
                        color = eval(color)
                elif line.startswith('plot_linestyle'):
                    lstyle = line[len('plot_linestyle,'):]
                    if lstyle.startswith('('):
                        lstyle = eval(lstyle)
                start_idx += 1

        df = pd.read_csv(log_filename, skiprows=range(start_idx))  # dataframe starts from start_idx
        values = df[df['epoch'] == -1][['round', metric]].sort_values(['round'])[metric].tolist()
        if niters:
            values = values[:niters]
        
        # plot global loss for each configuration
        if plot_label:
            p_label = plot_label
        elif algo =='rand' or algo =='adapow-d':
            p_label = algo
        else:
            p_label = algo+', d={}'.format(powd)
        plt.plot(values, lw=lw, color=color, ls = lstyle, label=p_label)

    # update plot settings
    plt.ylabel(f'Global {re.sub("_", " ", metric) + ("uracy" if metric.endswith("acc") else "")}')
    plt.xlabel('Communication round')
    plt.xticks()
    plt.yticks()
    loc = 'lower right' if metric.endswith('acc') else 'upper right'
    plt.legend(loc=loc)
    plt.grid()
    # plt.title('K=30, m={}'.format(clients_per_round))
    # plt.show()
    directory = os.path.dirname(log_filenames[0])
    plot_filename = os.path.join(directory, f'{name}_{metric}.pdf')
    if save:
        if save_path:
            plot_filename = save_path
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f'saving plot to {plot_filename}')
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make plots')
    parser.add_argument('-d', '--logdir', type=str, help='log directory')
    parser.add_argument('-m', '--metric', type=str, default='train_loss', help='metric to plot')
    args = parser.parse_args()

    log_filenames = glob(f'{args.logdir}/*.csv')

    # sort log filenames in custom order [rand, pow-d*, cpow-d*, rpow-d*, adapow-d*, afl]
    def sort_key(filename):
        filename = filename.split('/')[-1]
        if filename.startswith('rand'):
            return 0
        elif filename.startswith('pow-d'):
            return 1
        elif filename.startswith('cpow-d'):
            return 2
        elif filename.startswith('rpow-d'):
            return 3
        elif filename.startswith('adapow-d'):
            return 4
        elif filename.startswith('afl'):
            return 5
        else:
            return 6
    log_filenames.sort()
    log_filenames.sort(key=sort_key)

    make_plot(log_filenames, args.metric)