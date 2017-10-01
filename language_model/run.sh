CUDA_VISIBLE_DEVICES=${1} python train_lm.py --train train.txt --dev valid.txt --test test.txt --depth $2 --d 500 --omer --content $3 --gates $4 > logs/L${2}.${3}.${4}
