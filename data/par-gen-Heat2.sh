#!/bin/sh

# trap "kill 0" EXIT

# python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=1 &
# python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=2 &
# python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=3 &
# python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=4 &
# python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=5 &
# python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=6 &
# python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=7 &
# python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=8 &
# python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=9 &
# python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=10 &
# python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=11 &
# python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=12 &
# python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=13 &
# python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=14 &
# python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=15 &
# python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=16 &
# python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=17 &
# python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=18 &
# python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=19 &
# python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=20 &
# python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=21 &
# python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=22 &
# python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=23 &
# python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=24 &
# python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=25 &
# python generate.py -domain='Heat2' -Ntrain=[5,2] -Ntest=[512,512] -trial=26 &
# python generate.py -domain='Heat2' -Ntrain=[5,2] -Ntest=[512,512] -trial=27 &
# python generate.py -domain='Heat2' -Ntrain=[5,2] -Ntest=[512,512] -trial=28 &
# python generate.py -domain='Heat2' -Ntrain=[5,2] -Ntest=[512,512] -trial=29 &
# python generate.py -domain='Heat2' -Ntrain=[5,2] -Ntest=[512,512] -trial=30 &

# wait

trap "kill 0" EXIT

python generate.py -domain='Heat2' -Ntrain=[20,5] -Ntest=[512,512] -trial=1 &
python generate.py -domain='Heat2' -Ntrain=[10,5] -Ntest=[512,512] -trial=7 &
python generate.py -domain='Heat2' -Ntrain=[20,2] -Ntest=[512,512] -trial=11 &
python generate.py -domain='Heat2' -Ntrain=[10,2] -Ntest=[512,512] -trial=16 &
python generate.py -domain='Heat2' -Ntrain=[20,10] -Ntest=[512,512] -trial=21 &

wait