# # run fedavg with client selection
# # h0,h1,.. are hosts names of the clusters
# # replace them either by IP address or your own host names

# pdsh -R ssh -w h0,h1,h2 "pkill python3.5"

# pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
#                         --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype rand \
#                         --powd 2 --ensize 100 --fracC 0.03 --size 3 \
#                         --save -p --optimizer fedavg --model MLP\
#                         --rank %n  --backend nccl --initmethod tcp://h0 \
#                         --rounds 300 --seed 2 --NIID --print_freq 50"

# pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
#                         --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
#                         --powd 6 --ensize 100 --fracC 0.03 --size 3 \
#                         --save -p --optimizer fedavg --model MLP\
#                         --rank %n  --backend nccl --initmethod tcp://h0 \
#                         --rounds 300 --seed 2 --NIID --print_freq 50"

# # command for single machine
# python train_dnn.py \
#     --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype rand \
#     --powd 2 --ensize 100 --fracC 0.03 \
#     --save -p --optimizer fedavg --model MLP \
#     --rounds 300 --seed 2 --NIID --print_freq 50 \
#     --rank 0 --size 1

# ===============================
# Figure 4: (a), seed = {1, 2, 3}, alpha = {2, 0.3}, client_sel_algo = {rand, pow-d6, pow-d9, pow-d15}
python train_dnn.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype rand \
    --num_clients 100 --fracC 0.03 \
    --save -p --optimizer fedavg --model MLP \
    --rounds 500 --seed 1 --NIID --print_freq 1 \
    --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29500

python train_dnn.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
    --powd 6 --num_clients 100 --fracC 0.03 \
    --save -p --optimizer fedavg --model MLP \
    --rounds 500 --seed 1 --NIID --print_freq 1 \
    --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29501

python train_dnn.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
    --powd 9 --num_clients 100 --fracC 0.03 \
    --save -p --optimizer fedavg --model MLP \
    --rounds 500 --seed 1 --NIID --print_freq 1 \
    --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29502

python train_dnn.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
    --powd 15 --num_clients 100 --fracC 0.03 \
    --save -p --optimizer fedavg --model MLP \
    --rounds 500 --seed 1 --NIID --print_freq 1 \
    --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29503


# =============== latest ===============
python train_dnn.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --seltype rand \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 500 --seed 1 