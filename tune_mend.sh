export CUDA_VISIBLE_DEVICES=$1
lrs=$2

# 5e-05 1e-05 5e-06 1e-06 5e-07 1e-07
for lr in $lrs
do
    python dev-mend.py --edit_lr ${lr}
done