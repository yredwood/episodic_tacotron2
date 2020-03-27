cmd1=$1
cmd2=$2




name=mellotron_warmup_nof0nosp
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py --hparams=distributed_run=True \
    --output_directory=models/$name --log_directory=logs/$name \
    -c models/mellotron_libritts.pt --warm_start

#CUDA_VISIBLE_DEVICES=1 python train.py --hparams=distributed_run=False \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_libritts.pt --warm_start
