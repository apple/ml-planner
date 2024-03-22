export PYTHONPATH="${PYTHONPATH}:text_autoencoder"

B=16
diffusion_task=demo
iteration=001 
CFG=5
step=20
ae_path=models/ae/

##
head -n $B $(dirname $DATA)/source_dev.txt > $(dirname $DATA)/source_dev_$B.txt
head -n $B $(dirname $DATA)/target_dev.txt > $(dirname $DATA)/target_dev_$B.txt

# generate latent code
sample_dir=samples/${diffusion_task}/step${step}
mkdir -p ${sample_dir}
model_path=models/diffusion/${diffusion_task}
ckpt_name=model000${iteration}K.pt
SEED=137
DATA=data-bin/dummy_sum_data/parsed_raw_pre/dev/dummy.pt

python neural_diffusion/sample_repr.py \
    --config-name sample_config \
    hydra.searchpath=[$model_path/config] \
    model_config=user \
    model_path=$model_path/$ckpt_name \
    num_samples=$B batch_size=$B \
    sample_dir=$sample_dir \
    output_full_traj=True \
    seed=$SEED \
    data_dir=$DATA \
    cond=0 \
    +load_ckpt=$ae_path \
    model_config.diffusion.ddpm_ddim_eta=0 \
    model_config.diffusion.continuous_steps=${step} \
    ++model_config.model.class_cond=True \
    ++class_cond=True \
    forward=False \
    use_ddim=True \
    store_only_last_sample=False \
    cfg_weight=${CFG} \
    use_cross_attention=True \
    ++print_dataset=False \
    ++input_text=$(dirname $DATA)/source_dev_$B.txt
   
# generate text from the latent code
filename=${sample_dir}/samples_*_ddim.npz

python text_autoencoder/gen_from_repr.py \
--load_ckpt $ae_path \
--num_feature 16 --sentence_len 256 \
--remove_dash_n \
--cond -1 \
--sample_size $B \
--repr_file ${filename} \


## Evaluation
grep -r "^Time 0\.00:" ${sample_dir}/samples_*_ddim.top1.txt| sed 's/samples\/\([^\/]*\)\/.*:\(.*\)/\1\t\2/' | sed 's/Time 0.00://g'> ${sample_dir}/${diffusion_task}_generation_${step}_cfg_${CFG}_${SEED}.txt
mkdir -p ${sample_dir}/eval
mv ${sample_dir}/${diffusion_task}_generation_*.txt ${sample_dir}/eval
python text_autoencoder/eval_ground.py --ref $(dirname ${DATA})/target_dev_$B.txt --eval ${sample_dir}/eval


## Evaluation: compute AuBLEU
# diffusion_task=pu27kqj9ji 
# iteration=061
# # python neural_diffusion/get_model.py -t ${diffusion_task} -i ${iteration}
# step=5

# model_path=models/diffusion/${diffusion_task}
# ckpt_name=model000${iteration}K.pt
# ae_path=models/ceb2fmf4jv
# -m debugpy --listen 0.0.0.0:5678 --wait-for-client 
python neural_diffusion/sample_repr.py \
    --config-name sample_config \
    hydra.searchpath=[$model_path/config] \
    model_config=user \
    model_path=$model_path/$ckpt_name \
    num_samples=$B batch_size=$B \
    sample_dir=$sample_dir \
    output_full_traj=True \
    seed=$SEED \
    data_dir=$DATA \
    cond=0 \
    +load_ckpt=$ae_path \
    model_config.diffusion.ddpm_ddim_eta=0 \
    model_config.diffusion.continuous_steps=${step} \
    ++model_config.model.class_cond=True \
    ++class_cond=True \
    forward=True \
    use_ddim=True \
    store_only_last_sample=False \
    cfg_weight=${CFG} \
    use_cross_attention=True \
    ++print_dataset=False \
    ++input_text=$(dirname $DATA)/source_dev_$B.txt


python text_autoencoder/gen_from_repr.py \
--load_ckpt $ae_path \
--num_feature 16 --sentence_len 256 \
--remove_dash_n \
--forward \
--cond -1 \
--sample_size $B \
--repr_file ${sample_dir}/forward_*.npz \
--dev_pt_dir $(dirname ${DATA})/target_dev_$B.txt \

