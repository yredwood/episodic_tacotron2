cmd1=$1
cmd2=$2




name=mellotron_warmup
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m multiproc train.py --hparams=distributed_run=True \
    --output_directory=models/$name --log_directory=logs/$name \
    -c models/mellotron_libritts.pt --warm_start

