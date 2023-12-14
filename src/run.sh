#!/bin/bash
python train.py ../models/lindsey_weight_reg_act/l2reg_leaky cifar10 32 0.01 --optim="msgdw" --weight_decay=2e-4 --weight_norm=2 --early_stop=10 --save_hist

python train.py ../models/lindsey_act/color_ks5_leaky cifar10 128 0.0001 --optim="sgd"