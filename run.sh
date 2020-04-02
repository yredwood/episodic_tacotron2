cmd1=$1
cmd2=$2




name=mellotron_bugfix_autoencoder_singleh_tf0.0
CUDA_VISIBLE_DEVICES=6,7 python -m multiproc train.py --hparams=distributed_run=True \
    --output_directory=models/$name --log_directory=logs/$name \

#CUDA_VISIBLE_DEVICES=4,5 python train.py --hparams=distributed_run=False \
#    --output_directory=models/$name --log_directory=logs/$name \
