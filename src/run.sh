#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py ../models/alexnet_optim/alexnet_sgd_wdecay00001 cifar10 32 0.01 --optim=sgd --weight_decay=1e-4 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/alexnet_optim/alexnet_sgd_wdecay0000001 cifar10 32 0.01 --optim=sgd --weight_decay=1e-6 --momentum=0.5 --early_stop=10 --save_hist

CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/lindseydefaultgrey cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/lindsey32 cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/lindsey32grey cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/color_ks3 cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/color_ks5 cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/lindsey32gelu cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist
CUDA_VISIBLE_DEVICES=0 python train.py ../models/lindsey_optim/lindseydefaultgelu cifar10 32 0.01 --optim=sgd --weight_decay=1e-3 --momentum=0.5 --early_stop=10 --save_hist