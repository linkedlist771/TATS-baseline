

1. Tran VQGan

```bash 
export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
                      --gpus 1 --sync_batchnorm --batch_size 2 --num_workers 32 --accumulate_grad_batches 6 \
                      --progress_bar_refresh_rate 500 --max_steps 2000000 --gradient_clip_val 1.0 --lr 3e-5 \
                      --data_path /home/gpu02/dingli/VideoGPT-baseline/data/deposition_data_video_split/ --default_root_dir output \
                      --resolution 384 --sequence_length 8 --discriminator_iter_start 10000 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

```