CUDA_VISIBLE_DEVICES=${1} python -u train_lm.py --train train.txt --dev valid.txt --test test.txt --depth $2 --d 500 --omer --content $3 --gates $4 > logs_content/L${2}.${3}.${4}
