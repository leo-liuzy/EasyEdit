export CUDA_VISIBLE_DEVICES=3

python run_zsre_llama2.py \
    --editing_method=ROME \
    --data_dir=../data \
    --hparams_dir=../hparams/ROME/llama-7b \
    # --hparams_dir=../hparams/ROME/llama3-8b \