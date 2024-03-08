CUDA_VISIBLE_DEVICES=$1
CONFIG=$2
DATA_ROOT=$3
USE_CHECKPOINT=$4

PYTHONPATH=$PWD:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -u tools/train.py $CONFIG \
  --cfg-options train_dataloader.dataset.data_root=$DATA_ROOT \
  --cfg-options val_dataloader.dataset.data_root=$DATA_ROOT \
  --cfg-options test_dataloader.dataset.data_root=$DATA_ROOT \
  --cfg-options model.backbone.use_checkpoint=$USE_CHECKPOINT

# bash ./tools/train.sh 1 ./configs/transmorph/transmorph_swin_ixi-160x192x224.py /home/hb/public/i2i/datasets/IXI_data/ True
