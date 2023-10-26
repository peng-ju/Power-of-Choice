import numpy as np
import numpy.random as rv
import numpy.linalg as lin
import matplotlib.pyplot as plt
from quadoption import args_parser

args = args_parser()

def get_rhos(p_dat, it):
    '''
    Get the estimate rhos (not theoretical ones) by setting the range of $w$ as the $\overbar{w}$s for each iteration
    :return: estimate of rhobar, rhotilde
    
    produce idxs_users, which is the selected clients for next iteration
    '''
    size = args.num_users*args.frac
    selidxs_freq = np.zeros(args.num_users)
    p_dat = np.ndarray.flatten(p_dat)

    for ii in range(it + 1):
        if args.seltype == 'rand':
            idxs_users = np.random.choice(args.num_users, p=p_dat, size=int(size), replace=True)
        elif args.seltype == 'powd':
            rnd_idx = np.random.choice(args.num_users, size=args.powd, replace=False, p=p_dat)
            repval = list(zip([currcli_loss[i] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))
            # select top [size] clients based on the loss
            idxs_users = rep[1][:int(size)]

        if ii != it:
            for j in idxs_users:
                selidxs_freq[j] += 1

    # Get rho values
    rhobar = np.inf
    for i in range(args.le):
        le_currloss = callocloss(np.ndarray.flatten(np.transpose(x_temp[i])), all_mult_mat, x_mult_opt)
        le_numi = (sum(np.multiply(selidxs_freq/it, np.array(le_currloss)-np.array(loc_fmin))))\
                  /(args.frac*args.num_users)
        le_loss = (np.sum(np.multiply(le_currloss, p_dat)) - glob_loss)\
                  -sum(np.multiply(p_dat,loc_fmin)) # Global loss F(w^i)
        if rhobar>=le_numi/le_loss:
            rhobar = le_numi/le_loss

    rhotilde_numi = sum(np.multiply(selidxs_freq/it, np.array(local_optloss)-np.array(loc_fmin)))/\
                    (args.frac*args.num_users)
    rhotilde = rhotilde_numi / rhotilde_deno

    return idxs_users, rhobar, rhotilde


def callocloss(x,H,e):
    """
    equation (87) of the supplementary material
    L = (Hx - e)**2 
    
    x: current parameters
    H: fixed matrix for the quadratic objective function
    e: fixed vector
    """
    ans = []

    if len(np.shape(x))==1 or np.shape(x)[0]==1:
        for i in range(args.num_users):
            term1 = np.matmul(np.matmul(x, H[i]), np.transpose(x)) / 2
            term2 = -np.matmul(x, np.transpose(e[i][:]))
            term3 = np.matmul(np.matmul(e[i][:], lin.inv(H[i])), np.transpose(e[i][:])) / 2

            ans.append(term1+term2+term3)

    else:
        for i in range(args.num_users):
            term1 = np.matmul(np.matmul(x[i][:], H[i]), np.transpose(x[i][:])) / 2
            term2 = -np.matmul(x[i][:], np.transpose(e[i][:]))
            term3 = np.matmul(np.matmul(e[i][:], lin.inv(H[i])), np.transpose(e[i][:])) / 2

            ans.append(term1+term2+term3)

    return ans


if __name__ == '__main__':

    # num_users: number of clients
    # frac: C fraction of client is chosen for each iteration
    size = args.num_users*args.frac 
    rnd = rv.RandomState(args.seed)
    
    # optimum result for each local client: number of client * dimension of data
    x_mult_opt = rnd.random_sample((args.num_users,args.dim))*100

    # array to produce H_k
    x_mult_mat = rnd.random_sample(args.num_users)*args.high
    
    all_mult_mat = {}
    loc_min, loc_fmin = [], []

    for k in range(args.num_users):
        # create diagonal matrix H_k = h_k * I
        all_mult_mat[k] = np.diag(np.full(args.dim,x_mult_mat[k]))
        # exact local minimum: inv(H_k) * e_k for equation (87)
        loc_min.append(np.matmul(x_mult_opt[k][:], lin.inv(all_mult_mat[k])))   # x_k^*

    # Get Datasize
    # produce a 1d array: probability of each client to be choosen
    if args.eq == 1:
        p_dat = np.ones(args.num_users) / args.num_users

    else:
        p_dat = rnd.power(args.alpha, args.num_users)
        p_dat = p_dat/sum(p_dat)

    opt1 = np.sum(p_dat * np.array(x_mult_mat))
    opt2 = np.matmul(p_dat.T, x_mult_opt)
    # globle minimum
    glob_min = np.matmul(opt2, lin.inv(np.diag(np.full(args.dim,opt1))))      # x^*


    local_optloss = callocloss(glob_min, all_mult_mat, x_mult_opt)
    loc_fmin = callocloss(loc_min, all_mult_mat, x_mult_opt)
    glob_loss = np.sum(np.multiply(p_dat, local_optloss))

    disto_glob, x_global = [], []
    cli_loss, currcli_loss, global_loss = [], [], []
    rhobar_dat, rhotilde_dat = [], []
    rhotilde_deno = sum(np.multiply(local_optloss,p_dat))-sum(np.multiply(loc_fmin,p_dat))

    # initilize gobal parameter: w
    x_global.append(rnd.randn(1, args.dim)*100)

    currcli_loss = callocloss(np.ndarray.flatten(np.transpose(x_global[0][:])),all_mult_mat,x_mult_opt)

    cli_loss.append(currcli_loss)
    loss = np.sum(np.multiply(currcli_loss, p_dat))-glob_loss                          # Global loss F(w^i)
    global_loss.append(loss)

    # record selected frequency for each clients
    sel_freq = np.zeros(args.num_users)
    all_sel_freq = []

    lr = args.lr

    new_global = 0
    for epoch in range(args.epochs):
            # Select users
            if epoch ==0:           # If it is the first epoch so we first randomly select the users by their data size
                init = np.random.RandomState(1)
                if args.seltype == 'rand': # why seltype change replace or not
                    idxs_users = init.choice(args.num_users, p=p_dat, size=int(size), replace=True)

                else:
                    idxs_users = init.choice(args.num_users, p=p_dat, size=int(size), replace=False)

            '''
            if epoch==4000 or epoch==8000:
                lr /= 4
                print('update learning rate to', args.lr)
            '''
            disto_glob.append(lin.norm(glob_min-x_global[epoch], ord=2))

            # For Selected Clients: local update (epoch, batch) & data accum.
            # le: local updates for each client
            x_temp = {k:np.zeros(np.shape(x_global[epoch][:])) for k in range(int(args.le))}
            for j, idx in enumerate(idxs_users):
                idx = int(idx)
                tmp = x_global[epoch][:]
                for i in range(args.le):
                    tmp = tmp - lr * (np.matmul(tmp,all_mult_mat[idx]) - x_mult_opt[idx][:])
                    x_temp[i]+=tmp

                    # average of all selected clients
                    if j == int(size)-1:
                        x_temp[i]/=size

                sel_freq[idx]+=1

            all_sel_freq.append(list(sel_freq))
            # only take the result from the latest local update
            x_global.append(x_temp[args.le-1]) 

            currcli_loss = callocloss(np.ndarray.flatten(np.transpose(x_global[epoch+1][:])), all_mult_mat, x_mult_opt)
            cli_loss.append(currcli_loss)
            loss = np.min(np.sum(np.multiply(currcli_loss, p_dat))-glob_loss,0)        # Global loss F(w^i)
            global_loss.append(loss)

            # select the clients for next epoch
            idxs_users, rhobar, rhotilde = get_rhos(p_dat, it=100)
            rhobar_dat.append(rhobar)
            rhotilde_dat.append(rhotilde)

            # print("idxs_users", idxs_users, "rhobar", rhobar, "rhotilde", rhotilde)
            print('Epoch', epoch, 'Loss', loss, 'Distance to global minimum', lin.norm(glob_min-x_global[epoch], ord=2))

    plt.figure()
    plt.plot(global_loss)
    plt.ylabel('Global loss')
    plt.xlabel('Communication round')
    legend_properties = {'weight':'bold'}
    plt.xticks()
    plt.yticks()
    # plt.legend(loc=1)
    plt.grid()
    # plt.title('K=30, m={}'.format(sample_ratio))
    plt.show()
    plt.savefig("test.png")

    print('estimate rhobar', np.min(rhobar_dat))
    print('estimate rhot/rhob', np.max(rhotilde_dat)/np.min(rhobar_dat))


