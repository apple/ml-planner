EXP=demo
DATE_WITH_TIME=`date "+%m%d-%H%M%S"`

# run the experiments
# export PYTHONPATH="${PYTHONPATH}:/workspace/"
export OPAL_PREFIX=
export DISABLE_ZERO_MODULE=0

# CUDA_VISIBLE_DEVICES=0 
export PYTHONPATH="${PYTHONPATH}:text_autoencoder"
# 
python neural_diffusion/train.py \
    data_dir=data-bin/dummy_sum_data/parsed_raw_pre/train \
    model_config/model=cond_dit \
    model_config/diffusion=p2_x0_cont \
    trainer.lr=1e-4 \
    class_cond=True \
    random_flip=False \
    batch_size=12 \
    use_fp16=True \
    log_dir=models/diffusion/${EXP}_${DATE_WITH_TIME} \
    trainer.save_interval=1000 \
    model_config.model.num_res_blocks=1 \
    model_config.model.in_channels=1024 \
    model_config.model.out_channels=1024 \
    model_config.model.num_channels=256 \
    trainer.cfg_dropout=0.1 \
    load_ckpt=models/ae/ \
    +trainer.use_cross_attention=True \
    +model_config.diffusion.linear_beta=False \
    model_config.model.cond_model=t5-large \
    model_config.model.cond_finetune=True \
    model_config.model.cond_dim=64