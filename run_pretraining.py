# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import wandb

from argparse import ArgumentParser
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from hb_ssl.default_params import *
from hb_ssl.nn import FPN3d
from hb_ssl.pretrain.model import Vox2Vec
from hb_ssl.pretrain.pretraining_dataset import PretrainingDataset


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--pretraining_dataset', default='nako')
    parser.add_argument('--log_dir', default='/path/to/output_dir/')

    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)
    parser.add_argument('--pretrain_batch_size', type=int, default=10)
    parser.add_argument('--pretrain_num_workers', type=int, default=8)
    parser.add_argument('--num_batches_per_epoch', type=int, default=100)

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()


def main(args):

    patch_size = tuple(args.patch_size)

    pretrain_dataset = PretrainingDataset(
        patch_size=patch_size,
        max_num_voxels_per_patch=MAX_NUM_VOXELS_PER_PATCH,
        batch_size=args.pretrain_batch_size,
        batches_per_epoch=args.num_batches_per_epoch,
        pretraining_dataset=args.pretraining_dataset
    )

    pretrain_dataloader = DataLoader(
        dataset=pretrain_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=args.pretrain_num_workers
    )

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    model = Vox2Vec(
        backbone=backbone,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
    )

    checkpoint_callback_2 = ModelCheckpoint(dirpath=args.log_dir, save_top_k=5, monitor='epoch', mode='max', every_n_epochs=100, filename='{epoch:02d}')

    run = wandb.init(project='ssl', name='nakoregion2_fpn')
    logger = WandbLogger(experiment=run)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback_2],
        accelerator='gpu',
        max_epochs=500,
        gradient_clip_val=1.0
    )
    trainer.fit(
        model=model,
        train_dataloaders={
            'pretrain': pretrain_dataloader,
        },
    )


if __name__ == '__main__':
    main(parse_args())
