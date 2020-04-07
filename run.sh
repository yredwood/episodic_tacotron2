cmd1=$1
cmd2=$2





name=__from_scratch_tanh_additive_stf1.0
#CUDA_VISIBLE_DEVICES=2,3 python -m multiproc train.py --hparams=distributed_run=True \
    #--output_directory=models/$name --log_directory=logs/$name 

CUDA_VISIBLE_DEVICES=4 python train.py --hparams=distributed_run=False \
    --output_directory=models/$name --log_directory=logs/$name 
