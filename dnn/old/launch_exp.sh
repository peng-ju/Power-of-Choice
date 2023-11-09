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

# # ==================ORIGINAL DNN=============
# # Figure 4: (a), seed = {1, 2, 3}, alpha = {2, 0.3}, client_sel_algo = {rand, pow-d6, pow-d9, pow-d15}
# python train_dnn.py \
#     --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype rand \
#     --num_clients 100 --fracC 0.03 \
#     --save -p --optimizer fedavg --model MLP \
#     --rounds 500 --seed 1 --NIID --print_freq 1 \
#     --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29500

# python train_dnn.py \
#     --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
#     --powd 6 --num_clients 100 --fracC 0.03 \
#     --save -p --optimizer fedavg --model MLP \
#     --rounds 500 --seed 1 --NIID --print_freq 1 \
#     --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29501

# python train_dnn.py \
#     --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
#     --powd 9 --num_clients 100 --fracC 0.03 \
#     --save -p --optimizer fedavg --model MLP \
#     --rounds 500 --seed 1 --NIID --print_freq 1 \
#     --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29502

# python train_dnn.py \
#     --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
#     --powd 15 --num_clients 100 --fracC 0.03 \
#     --save -p --optimizer fedavg --model MLP \
#     --rounds 500 --seed 1 --NIID --print_freq 1 \
    --rank 0 --size 3 --backend gloo --initmethod tcp://localhost:29503


# =============== latest ===============
# Figure 4: (a), seed = {1, 2, 3}, alpha = {2, 0.3}, client_sel_algo = {rand, pow-d6, pow-d9, pow-d15}
python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo rand \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 400 --seed 2 --name fig4a

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 6 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 400 --seed 1 --name fig4a

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 9 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 400 --seed 1 --name fig4a

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 15 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 400 --seed 1 --name fig4a

## 4b
python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 0.3 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo rand \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 100 --seed 1 --name fig4b

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 0.3 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 6 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 100 --seed 1 --name fig4b

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 0.3 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 9 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 100 --seed 1 --name fig4b

python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 0.3 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 15 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 100 --seed 1 --name fig4b

# line profiler
# 1. rand
python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo rand \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 5 --seed 1 

kernprof -l trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo rand \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 5 --seed 1 

python -m line_profiler trainer.py.lprof > trainer_rand.lprof.txt

# 2. powd9
python trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 9 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 5 --seed 1

kernprof -l trainer.py \
    --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 \
    --dataset fmnist --NIID --print_freq 1 \
    --model MLP --algo pow-d --powd 9 \
    --num_clients 100 --clients_per_round 3 \
    --save -p --rounds 5 --seed 1

python -m line_profiler trainer.py.lprof > trainer_powd9.lprof.txt