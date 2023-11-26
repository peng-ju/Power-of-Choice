# run fedavg with client selection
# h0,h1,.. are hosts names of the clusters
# replace them either by IP address or your own host names

pdsh -R ssh -w h0,h1,h2 "pkill python3.5"

pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
                        --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype rand \
                        --powd 2 --ensize 100 --fracC 0.03 --size 3 \
                        --save -p --optimizer fedavg --model MLP\
                        --rank %n  --backend nccl --initmethod tcp://h0 \
                        --rounds 300 --seed 2 --NIID --print_freq 50"

pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
                        --constantE --lr 0.005 --bs 64 --localE 30 --alpha 2 --dataset fmnist --seltype pow-d \
                        --powd 6 --ensize 100 --fracC 0.03 --size 3 \
                        --save -p --optimizer fedavg --model MLP\
                        --rank %n  --backend nccl --initmethod tcp://h0 \
                        --rounds 300 --seed 2 --NIID --print_freq 50"

