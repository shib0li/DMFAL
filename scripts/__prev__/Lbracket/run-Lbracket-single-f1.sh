#!/bin/bash

cd ../..

MAX_EPOCHS=1000
HORIZON=100
# DEVICE='cpu'
DEVICE='cuda:0'

# MAX_EPOCHS=10
# HORIZON=2
# DEVICE='cuda:0'

trap "kill 0" EXIT

python main-v2.py evaluation -heuristic 'UniDMFAL-F1' -domain 'Lbracket' -M 2  -trial 1 \
    -input_dim_list [2,2] -output_dim_list [2500,5776] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-4 -reg_strength 1e-3 \
    -opt_lr 1e-1 -T $HORIZON \
    -placement $DEVICE &
    
python main-v2.py evaluation -heuristic 'UniDMFAL-F1' -domain 'Lbracket' -M 2  -trial 2 \
    -input_dim_list [2,2] -output_dim_list [2500,5776] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-4 -reg_strength 1e-3 \
    -opt_lr 1e-1 -T $HORIZON \
    -placement $DEVICE &
    
python main-v2.py evaluation -heuristic 'UniDMFAL-F1' -domain 'Lbracket' -M 2  -trial 3 \
    -input_dim_list [2,2] -output_dim_list [2500,5776] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-4 -reg_strength 1e-3 \
    -opt_lr 1e-1 -T $HORIZON \
    -placement $DEVICE &
    
    
python main-v2.py evaluation -heuristic 'UniDMFAL-F1' -domain 'Lbracket' -M 2  -trial 4 \
    -input_dim_list [2,2] -output_dim_list [2500,5776] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-4 -reg_strength 1e-3 \
    -opt_lr 1e-1 -T $HORIZON \
    -placement $DEVICE &
    
python main-v2.py evaluation -heuristic 'UniDMFAL-F1' -domain 'Lbracket' -M 2  -trial 5 \
    -input_dim_list [2,2] -output_dim_list [2500,5776] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,3] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-4 -reg_strength 1e-3 \
    -opt_lr 1e-1 -T $HORIZON \
    -placement $DEVICE &
    
wait