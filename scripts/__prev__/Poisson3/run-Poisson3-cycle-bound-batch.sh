#!/bin/sh

cd ../..

MAX_EPOCHS=1000
BATCH_SIZE=20
HORIZON=20
DEVICE='cpu'

# MAX_EPOCHS=10
# BATCH_SIZE=4
# HORIZON=2
# DEVICE='cpu'


trap "kill 0" EXIT

python main.py evaluation -heuristic 'CycleBoundBatchDMFAL' -domain 'Poisson3' -M 3  -trial 1 \
    -input_dim_list [5,5,5] -output_dim_list [256,1024,4096] -base_dim_list [32,32,32] \
    -hlayers_w [40,40,40] -hlayers_d [2,2,2] \
    -activation 'tanh' -penalty [1,2,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE &
    
python main.py evaluation -heuristic 'CycleBoundBatchDMFAL' -domain 'Poisson3' -M 3  -trial 2 \
    -input_dim_list [5,5,5] -output_dim_list [256,1024,4096] -base_dim_list [32,32,32] \
    -hlayers_w [40,40,40] -hlayers_d [2,2,2] \
    -activation 'tanh' -penalty [1,2,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE &
    
python main.py evaluation -heuristic 'CycleBoundBatchDMFAL' -domain 'Poisson3' -M 3  -trial 3 \
    -input_dim_list [5,5,5] -output_dim_list [256,1024,4096] -base_dim_list [32,32,32] \
    -hlayers_w [40,40,40] -hlayers_d [2,2,2] \
    -activation 'tanh' -penalty [1,2,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE &
    
python main.py evaluation -heuristic 'CycleBoundBatchDMFAL' -domain 'Poisson3' -M 3  -trial 4 \
    -input_dim_list [5,5,5] -output_dim_list [256,1024,4096] -base_dim_list [32,32,32] \
    -hlayers_w [40,40,40] -hlayers_d [2,2,2] \
    -activation 'tanh' -penalty [1,2,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE &
    
python main.py evaluation -heuristic 'CycleBoundBatchDMFAL' -domain 'Poisson3' -M 3  -trial 5 \
    -input_dim_list [5,5,5] -output_dim_list [256,1024,4096] -base_dim_list [32,32,32] \
    -hlayers_w [40,40,40] -hlayers_d [2,2,2] \
    -activation 'tanh' -penalty [1,2,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE &
    

wait
