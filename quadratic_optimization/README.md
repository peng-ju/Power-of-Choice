## Directory Structure
```
.
├── quadratic_sim.ipynb                 # Quadratic optimization jupyter notebook
├── quadratic_sim_MLflow.ipynb          # Quadratic optimization jupyter notebook. Use MLflow to tracking training metric and hyperparameters.
├── quadoption.py                       # args_parser for quadratic_sim.py
├── quadratic_sim.py                    # python script for Quadratic optimization
├── figures        
└── README.md
```

### Problem formulation for quadratic model optimization

1, The local objective function/loss function of k-th client (out of totally K clients) is:

$F_k(w) = 1/2 w^{T} H_k w - e^T_k w + 1/2 e^T_k H^{-1} e_k$

For simplicity, $H_k = h_k I$ is a diagonal matrix and $e_k$ is an arbitrary vector. The optimum for $F_k(w)$ is $w^* = H^{-1}_k e_k = e_k / h_k$. 

Gradient descent update rule for local model of k-th client is:

$w^{(t+1)}_k = w^{(t)}_k - \eta (H_k w^{(t)}_k - e_k)$



2, Global objective function/loss function 

$F(w) = \sum^K_{k=1} p_k F_k(w)$

$p_k$ represents the data size for the k-th client, which follows the power function distribution.

The optimum for global objective function $F(w)$ is:

$w^* = (\sum^K_{k=1} p_k H_k)^{-1} (\sum^K_{k=1} e_k) = (\sum^K_{k=1} e_k) / (\sum^K_{k=1} p_k h_k)$.



3, For each communication round, we sample $m = int(C * K)$ clients and takes the average of local models to calculate global model: $\bar{w}^{t+1} = 1/m \sum_{k \in S^t} w^{t+1}_k$

Each clients takes $\tau$ gradient descent update with a fixed learning rate $\eta$ before update to server/global model.

## Implementation

- Try `quadratic_sim.ipynb` to implement quadratic optimization with different algorithms. 
- Use `quadratic_sim_MLflow.ipynb` if you want to record the training process on Dagshub.
- Note: it takes an average of 3 min to train 15000 epochs on a Intel i7-1370H CPU. Using MLflow to track the training metrix slows down the training by around 60 times, ending up taking 3 hours for the same training process.

### Hyperparameters

```
self.num_users = num_users # number of client
self.frac = 0.1
self.powd = powd # number of candidate for powd algorithm
self.alpha = 3 # hyperparameter for the data distribution
self.lr = 0.00002 # learning rate for statistical gradient descent.
self.le = 2 # local update iteration
self.epochs = epochs # num of communication rounds
self.seltype = seltype # client selection strategies/algorithms.
self.dim = 5 # dimension of the quadratic model
```
