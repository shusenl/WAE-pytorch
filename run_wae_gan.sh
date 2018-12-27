#### GAN on other dataset
python main.py --dataset other --model gan --max_epoch 250 --batch_size 100 \
    --lr 1e-3 --beta1 0.5 --beta2 0.999 --z_dim 64 --z_var 2 --reg_weight 100 \
    --dset_dir /path/to/file/list.txt