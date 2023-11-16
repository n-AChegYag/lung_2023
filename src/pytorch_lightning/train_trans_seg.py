import os
import json
import torch
import argparse
import lightning as L

import transforms

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    ModelSummary, 
    StochasticWeightAveraging, 
    LearningRateMonitor
)

from lung_dataset import LungDataset
from trainer_trans_seg import PLModule
import sys
sys.path.append('/home/acy/data/lung/src/')
from metrics import dice
from seg_loss.dice_loss import MySoftDiceLoss
from model.my_model import MyEncoder, MyDecoder

torch.backends.cudnn.benchmark = True
# L.seed_everything(426)
torch.set_float32_matmul_precision('high')

def load_pretrain_weights(pl_ckpt):
    from collections import OrderedDict
    new_model_state_dict = OrderedDict()
    for k,v in pl_ckpt['state_dict'].items():
        new_model_state_dict[k[6:]] = v
    return new_model_state_dict

def main(args):
    tag = f'{args.tag}_{args.n_filters}_{args.learning_rate}'
    patch_size = (32,144,144)
    precision = '16-mixed' if args.autocast else '32'
    # path_to_split_pkl = f'/home/acy/data/lung/src/PL/splits/split_cl_230817_{args.split}.pkl'
    path_to_split_pkl = '/home/acy/data/lung/src/PL/splits/splits_cl_231012_mc_c_16.pkl'
    pretrain_ckpt_path = '/home/acy/data/lung/log/231022_pre_cls_mc2_16_c_32_0.0001_0.5/lightning_logs/version_0/checkpoints/epoch=12-step=481.ckpt'
    pretrain_ckpt = torch.load(pretrain_ckpt_path)
    if args.resume:
        ckpt_path = ''

    # train, valid and test data paths
    with open(path_to_split_pkl) as f:
        splits = json.load(f)
        train_paths = splits['train']
        valid_paths = splits['valid']
        test_paths  = splits['test']

    # train and val data transforms:
    train_transforms = transforms.Compose([
        transforms.RandomRotation(p=0.5, angle_range=[0, 15]),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    # datasets
    train_set = LungDataset(
        train_paths,
        'train',
        train_transforms,
        patch_size
        )
    valil_set = LungDataset(
        valid_paths,
        'valid',
        val_transforms,
        patch_size
        )
    test_set = LungDataset(
        test_paths,
        'test',
        val_transforms,
        patch_size
        )
    
    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valil_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_set , batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model, trainer and pl_module
    encoder = MyEncoder(2,args.n_filters,2)
    decoder = MyDecoder(2,args.n_filters,2)
    encoder.load_state_dict(load_pretrain_weights(pretrain_ckpt), strict=False)
    trainer = L.Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'log', tag),
        devices=args.gpus,
        max_epochs=args.epochs,
        accelerator='gpu',
        strategy='auto',
        precision=precision,  
        log_every_n_steps=2,
        callbacks=[
            ModelSummary(),
            LearningRateMonitor('epoch'),
            ModelCheckpoint(monitor='valid_metric', save_top_k=3, mode='max'),
            StochasticWeightAveraging(swa_lrs=args.learning_rate/100, swa_epoch_start=0.8)
        ]
    )
    pl_kwargs = {
        'model':        {
            'E':        encoder,
            'D':        decoder,
        },
        'batch_size':   args.batch_size,
        'lr':           args.learning_rate,
        'optimizer':    torch.optim.Adam,
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'loss_fn':      MySoftDiceLoss(beta=1, smooth=1e-4),
        'metric_fn':    dice,
    }
    pl_coarse = PLModule(**pl_kwargs)

    # train, validation and test
    if args.mode == 'train':
        if args.resume:
            pl_coarse.load_from_checkpoint(ckpt_path, **pl_kwargs)
        trainer.fit(pl_coarse, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        pl_coarse = PLModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, **pl_kwargs)
    elif args.mode == 'test':
        pl_coarse.load_from_checkpoint(ckpt_path, **pl_kwargs)
    valid_result = trainer.test(pl_coarse, dataloaders=valid_loader, verbose=False)
    test_result  = trainer.test(pl_coarse, dataloaders=test_loader , verbose=False)
    print(f"validation metric: {valid_result[0]['test_metric']:.4f}")
    print(f"test metric: {test_result[0]['test_metric']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The parameters of the training')

    parser.add_argument("-t", "--tag", type=str, required=False, default='Latest',
                        help="Tag of this training")
    parser.add_argument("-g", "--gpus", nargs='+', type=int, required=False, default=[0],
                        help="Which GPUs are using in this training")
    parser.add_argument("-r", "--resume", type=bool, required=False, default=False,
                        help="Whether to continue training the model")
    parser.add_argument("-a", "--autocast", type=bool, required=False, default=False,
                        help="Whether to use autocast (torch.cuda.amp) during training")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=8,
                        help="Batch size of the training")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=500,
                        help="Number of epochs of the training")
    parser.add_argument("-w", "--workers", type=int, required=False, default=4,
                        help="Number of workers of the training")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=0.001,
                        help="Learning rate of the training")
    parser.add_argument("-f", "--n_filters",  type=int, required=False, default=32,
                        help="Number of filters of the training")
    parser.add_argument("-m", "--mode",  type=str, required=False, default='train',
                        help="Train, Test or Predict")
    parser.add_argument("-s", "--split",  type=str, required=False, default='16',
                        help="which splits")
    
    args = parser.parse_args()
    main(args)
