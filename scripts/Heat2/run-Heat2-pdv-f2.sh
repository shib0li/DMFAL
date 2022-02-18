#!/bin/bash

cd ../..

# MAX_EPOCHS=1000
# HORIZON=100
# DEVICE='cpu'

# MAX_EPOCHS=10
# HORIZON=2
# DEVICE='cuda:0'

MAX_EPOCHS=$1
HORIZON=$2
DEVICE=$3

echo $MAX_EPOCHS
echo $HORIZON
echo $DEVICE

trap "kill 0" EXIT

python main.py evaluation -heuristic='PdvDMFAL-F2' -domain='Heat2' -M=2  -trial=1 \
    -input_dim_list=[3,3] -output_dim_list=[256,1024] -base_dim_list=[32,32] \
    -hlayers_w=[40,40] -hlayers_d=[2,2] \
    -activation='tanh' -penalty=[1,3] \
    -print_freq=10 -max_epochs=$MAX_EPOCHS -learning_rate=1e-4 -reg_strength=1e-3 \
    -opt_lr=1e-1 -T=$HORIZON \
    -placement=$DEVICE &
   
python main.py evaluation -heuristic='PdvDMFAL-F2' -domain='Heat2' -M=2  -trial=2 \
    -input_dim_list=[3,3] -output_dim_list=[256,1024] -base_dim_list=[32,32] \
    -hlayers_w=[40,40] -hlayers_d=[2,2] \
    -activation='tanh' -penalty=[1,3] \
    -print_freq=10 -max_epochs=$MAX_EPOCHS -learning_rate=1e-4 -reg_strength=1e-3 \
    -opt_lr=1e-1 -T=$HORIZON \
    -placement=$DEVICE &
    
python main.py evaluation -heuristic='PdvDMFAL-F2' -domain='Heat2' -M=2  -trial=3 \
    -input_dim_list=[3,3] -output_dim_list=[256,1024] -base_dim_list=[32,32] \
    -hlayers_w=[40,40] -hlayers_d=[2,2] \
    -activation='tanh' -penalty=[1,3] \
    -print_freq=10 -max_epochs=$MAX_EPOCHS -learning_rate=1e-4 -reg_strength=1e-3 \
    -opt_lr=1e-1 -T=$HORIZON \
    -placement=$DEVICE &
    
python main.py evaluation -heuristic='PdvDMFAL-F2' -domain='Heat2' -M=2  -trial=4 \
    -input_dim_list=[3,3] -output_dim_list=[256,1024] -base_dim_list=[32,32] \
    -hlayers_w=[40,40] -hlayers_d=[2,2] \
    -activation='tanh' -penalty=[1,3] \
    -print_freq=10 -max_epochs=$MAX_EPOCHS -learning_rate=1e-4 -reg_strength=1e-3 \
    -opt_lr=1e-1 -T=$HORIZON \
    -placement=$DEVICE &
    
python main.py evaluation -heuristic='PdvDMFAL-F2' -domain='Heat2' -M=2  -trial=5 \
    -input_dim_list=[3,3] -output_dim_list=[256,1024] -base_dim_list=[32,32] \
    -hlayers_w=[40,40] -hlayers_d=[2,2] \
    -activation='tanh' -penalty=[1,3] \
    -print_freq=10 -max_epochs=$MAX_EPOCHS -learning_rate=1e-4 -reg_strength=1e-3 \
    -opt_lr=1e-1 -T=$HORIZON \
    -placement=$DEVICE &
  
wait