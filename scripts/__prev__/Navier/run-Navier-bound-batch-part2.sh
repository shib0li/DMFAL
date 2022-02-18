#!/bin/sh

cd ../..

MAX_EPOCHS=500
BATCH_SIZE=10
HORIZON=20
# DEVICE='cuda:0'
DEVICE='cpu'

# MAX_EPOCHS=10
# BATCH_SIZE=4
# HORIZON=2
# DEVICE='cpu'


# trap "kill 0" EXIT
   

python main.py evaluation -heuristic 'BoundBatchDMFAL' -domain 'Navier' -M 2  -trial 4 \
    -input_dim_list [5,5] -output_dim_list [54621,121296] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE 
    
python main.py evaluation -heuristic 'BoundBatchDMFAL' -domain 'Navier' -M 2  -trial 5 \
    -input_dim_list [5,5] -output_dim_list [54621,121296] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1.0 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE 
    
    
# wait
# wait
