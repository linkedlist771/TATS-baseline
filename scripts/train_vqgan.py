# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGAN, VideoData
from tats.modules.callbacks import ImageLogger, VideoLogger

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    # parser.add_argumentgument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--default_root_dir', type=str, default='checkpoints/vqgan')
    
    parser = VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    # automatically adjust learning rate
    bs, ngpu, accumulate = args.batch_size, args.gpus, args.accumulate_grad_batches
    base_lr = args.lr  # 现在 args.lr 来自 VQGAN.add_model_specific_args() 中定义的 3e-4
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    trainer_kwargs = {
        'callbacks': callbacks,
        'max_steps': args.max_steps,
        'gradient_clip_val': args.gradient_clip_val,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'default_root_dir': args.default_root_dir
    }

    if args.gpus > 1:
        trainer_kwargs.update({
            'accelerator': 'gpu',
            'devices': args.gpus,
            'strategy': 'ddp'
        })
    elif args.gpus == 1:
        trainer_kwargs.update({
            'accelerator': 'gpu',
            'devices': 1
        })

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

