cmd1=$1
cmd2=$2




name=gst_tacotron_original_stl_pps
#CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py --hparams=distributed_run=True \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 

CUDA_VISIBLE_DEVICES=2 python train.py --hparams=distributed_run=False \
    --output_directory=models/$name --log_directory=logs/$name 
