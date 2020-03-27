cmd1=$1
cmd2=$2





#name=mellotron_nof0
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m multiproc train.py --hparams=distributed_run=True \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_nof0/checkpoint_15500
nc=16
ns=16
nq=16
tfs_type=single

from=models/mellotron_libritts.pt
ws=true
#from=models/tst_tacotron2_161616_single_2_pretrained/checkpoint_18000

name="tst_tacotron2_${nc}${ns}${nq}_${tfs_type}_3"

hparams="num_common=$nc,num_support=$ns,num_query=$nq,tfs_type=$tfs_type"
params="CUDA_VISIBLE_DEVICES=$cmd2 python"
if [ $cmd1 = mult ]
then
    params="$params -m multiproc train.py"
    hparams="$hparams,distributed_run=True"
else
    params="$params train.py"
    hparams="$hparams,distributed_run=False"
fi

if [ ! -z ${from+x} ]
then
    name="${name}_pretrained"
    params="$params -c $from"
fi
if [ ! -z ${ws+x} ]
then
    echo "warm start set"
    params="$params --warm_start"
fi

params="$params --output_directory=models/$name --log_directory=logs/$name --hparams=$hparams"

echo $params
eval $params
#
#
#    CUDA_VISIBLE_DEVICES=$cmd2 python -m multiproc train.py \
#        --hparams=distributed_run=True \
#        --output_directory=models/$name --log_directory=logs/$name \
#        -c models/mellotron_libritts.pt --warm_start
#else
#    CUDA_VISIBLE_DEVICES=$cmd2 python train.py --hparams=distributed_run=False \
#        --output_directory=models/$name --log_directory=logs/$name \
#        -c models/$name/checkpoint_30000
#fi
#
#
#
#
#
#
#
#
#










#
