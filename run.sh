cmd1=$1
cmd2=$2




name=gst_tacotron_original_stl_pps_tf0.0_t
#CUDA_VISIBLE_DEVICES=4,5 python -m multiproc train.py --hparams=distributed_run=True \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 

CUDA_VISIBLE_DEVICES=0 python train.py --hparams=distributed_run=False \
    --output_directory=models/$name --log_directory=logs/$name 
