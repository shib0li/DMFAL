#!/bin/sh

trap "kill 0" EXIT

python generate.py -domain='Navier' -Ntrain=[20,5] -Ntest=[64,64] -trial=1 &
python generate.py -domain='Navier' -Ntrain=[20,5] -Ntest=[64,64] -trial=2 &
python generate.py -domain='Navier' -Ntrain=[20,5] -Ntest=[64,64] -trial=3 &
python generate.py -domain='Navier' -Ntrain=[20,5] -Ntest=[64,64] -trial=4 &
python generate.py -domain='Navier' -Ntrain=[20,5] -Ntest=[64,64] -trial=5 &

wait
