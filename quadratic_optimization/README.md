### Problem formulation for quadratic model optimization
Question: how to make this a ML problem?



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