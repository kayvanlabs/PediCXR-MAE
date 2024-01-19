# PediCXR-MAE
Self-Supervision and Transfer Learning for an Accurate Pediatric Chest X-Ray Diagnosis with Small Dataset

## Author: Yufeng Zhang (chloezh@umich.edu)

### example code for CNN (ResNet 50)
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

### example code for ViT
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main_finetune.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 128 \
    --finetune ${FINE_TUNE_FILE} \
    --epochs 75 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --model vit_base_patch16 \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 4 \
    --nb_classes 6 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1'\
    --seed 2
```

### example code for linear probing

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env main_linear_probe.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 128 \
    --finetune ${FINE_TUNE_FILE} \
    --epochs 75 \
    --blr 0.1 --weight_decay 0 \
    --model vit_base_patch16 \
    --cls_token \
    --dist_eval \
    --warmup_epochs 5 \
    --drop_path 0.2 \
    --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 4 \
    --nb_classes 6 \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
```
