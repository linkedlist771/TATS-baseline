

1. Tran VQGan


conda activate tats

```bash 
export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
                      --gpus 1 --sync_batchnorm --batch_size 2 --num_workers 32 --accumulate_grad_batches 6 \
                      --progress_bar_refresh_rate 500 --max_steps 2000000 --gradient_clip_val 1.0 --lr 3e-5 \
                      --data_path deposition_data_video_split/ --default_root_dir output \
                      --resolution 128 --sequence_length 8 --discriminator_iter_start 10000 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

```

2. Train transformer
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) &&  python scripts/train_transformer.py --num_workers 32 --val_check_interval 0.5 --progress_bar_refresh_rate 500 \
                        --gpus 1 --sync_batchnorm --batch_size 3 --unconditional \
                        --vqvae output/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt --data_path deposition_data_video_split/  --default_root_dir output_transformer \
                        --vocab_size 16384 --block_size 1024 --n_layer 24 --n_head 16 --n_embd 1024  \
                        --resolution 128 --sequence_length 8 --max_steps 2000000
```


3. Inference

- Sample for a small video:
> `--batch_size 1` means the there will be only one image per video. 
```bash
export CUDA_VISIBLE_DEVICES=1&&export PYTHONPATH=$PYTHONPATH:$(pwd) &&  python scripts/sample_vqgan_transformer_short_videos.py     --gpt_ckpt output_transformer/lightning_logs/version_2/checkpoints/best_checkpoint.ckpt --vqgan_ckpt output/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt     --save output_videos --batch_size 1 --resolution 128     --top_k 2048 --top_p 0.8 --save_videos

```

4. extract frames for eval

- sample for all just like the MoCoGan does:

```bash
python extract_frames_to_get_eval_dir.py --true_video_dir  deposition_data_video_split/test/ --generated_video_dir output_videos/videos/ucf101/topp0.80_topk2048_run0/
```

5. eval


In the SVD contianer:

```bash
CUDA_VISIBLE_DEVICES=3 && nohup python eval_metrics.py     --real-dir=/home/gpu02/dingli/TATS-baseline/extracted_frames/true/     --gen-dir=//home/gpu02/dingli/TATS-baseline/extracted_frames/generated/  > $(date +%m%d)"TATSFINtune".log 2>&1 &
```