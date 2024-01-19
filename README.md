# PediCXR-MAE
Self-Supervision and Transfer Learning for an Accurate Pediatric Chest X-Ray Diagnosis with Small Dataset


### example code
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main_finetune.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 128 \
    --finetune ${FINE_TUNE_FILE} \
    --checkpoint_type "smp_encoder" \
    --epochs 75 \
    --blr 2.5e-4 --weight_decay 0.05 \
    --model 'resnet50' \
    --warmup_epochs 5 \
    --drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 4 \
    --nb_classes 6 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1'
```
