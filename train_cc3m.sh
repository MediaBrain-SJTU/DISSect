# Example training script for CC3M dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun  --master_port 29510\
                                    --nproc_per_node 8 src/main.py \
                                    --train-data 'path/to/your/cc3m-train-{0000..0575}.tar' \
                                    --train-num-samples 2905954 \
                                    --dataset-type webdataset \
                                    --batch-size 1280 \
                                    --precision amp \
                                    --workers 8 \
                                    --val-data 'path/to/your/cc3m-validation-{0000..0015}.tar' \
                                    --lr 2e-3 \
                                    --warmup 10000 \
                                    --warmup-point 10 \
                                    --select-strategy 'warmup_base_sampling' \
                                    --select-rate 0.2 \
                                    --epochs 40 \
                                    --name 'ablation/sample_02_lr2e-3' \
                                    --select \
                                    --record
