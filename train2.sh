python train.py -d dataset --N 128 --M 192 --depth 2 0 2 0  --heads 4 --dim_head 192 --dropout 0.1 -e 100  -lr 1e-4 -n 8  --lambda 1e-2 --batch-size 4  --test-batch-size 4 --aux-learning-rate 1e-4 --patch-size 384 384 --cuda --save --seed 1926 --gpu-id 0 --savepath  ./checkpoint/V18_VRNoise --checkpoint ./checkpoint/PLConvTrans6/checkpoint_best_loss_45.pth.tar --training_stage 2 --stemode 0 --loadFromSinglerate 1