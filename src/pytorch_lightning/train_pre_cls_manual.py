import os
import json
import torch
import argparse
import random
import lightning as L

import transforms

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    ModelSummary, 
    StochasticWeightAveraging, 
    LearningRateMonitor
)
from torch import nn
from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score
from lung_dataset import LungDataset, LungRadiomicsDataset, LungClinicalDataset, LungRadiomicsClinicalDataset
from trainer_pre_cls import PLModule
from torchsampler import ImbalancedDatasetSampler
import sys
sys.path.append('/home/acy/data/lung/src/')
from monai.losses.focal_loss import FocalLoss
from metrics import accuracy
from model.my_model import MyEncoder, MyClassifer

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
# L.seed_everything(42)
torch.set_float32_matmul_precision('high')

class CrossEntropyFocalLoss(nn.Module):
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.focal_loss_fn = FocalLoss(to_onehot_y=True)
        self.alpha = alpha
        
    def forward(self, input, target):
        ce_loss = self.ce_loss_fn(input, target)
        focal_loss = self.focal_loss_fn(input, target)
        return self.alpha*ce_loss + (1-self.alpha)*focal_loss
    
def metric_fn(input, target, alpha=0.5):
    auc_fn = AUROC(task='binary')
    acc_fn = accuracy
    return alpha*auc_fn(input[:,1], target) + (1-alpha)*acc_fn(input, target)

def main(args):
    tag = f'{args.tag}_{args.split}_{args.feats}_{args.n_filters}_{args.learning_rate}_{args.alpha}'
    patch_size = (32,144,144)
    precision = '16-mixed' if args.autocast else '32'
    path_to_split_pkl = f'/home/acy/data/lung/src/PL/splits/splits_231012_mc_c_16.pkl'
    path_to_info = '/mnt/ssd2t/acy/lung_data_rearrange/for_cls_resample_clinical/info_1012.xlsx'
    if args.resume:
        ckpt_path = ''

    # train, valid and test data paths
    with open(path_to_split_pkl) as f:
        splits = json.load(f)
        train_paths = splits['train']
        random.shuffle(train_paths)
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
        transforms.ToTensor('test')
    ])

    # datasets
    if args.feats == 'r':
        train_set = LungRadiomicsDataset(
            train_paths,
            path_to_info,
            'train',
            train_transforms,
            patch_size,
            gt='manual'
            )
        valil_set = LungRadiomicsDataset(
            valid_paths,
            path_to_info,
            'valid',
            val_transforms,
            patch_size,
            gt='manual'
            )
        test_set = LungRadiomicsDataset(
            test_paths,
            path_to_info,
            'test',
            val_transforms,
            patch_size,
            gt='manual'
            )
    elif args.feats == 'c':
        train_set = LungClinicalDataset(
            train_paths,
            path_to_info,
            'train',
            train_transforms,
            patch_size,
            gt='manual'
            )
        valil_set = LungClinicalDataset(
            valid_paths,
            path_to_info,
            'valid',
            val_transforms,
            patch_size,
            gt='manual'
            )
        test_set = LungClinicalDataset(
            test_paths,
            path_to_info,
            'test',
            val_transforms,
            patch_size,
            gt='manual'
            )
    elif args.feats == 'cr':
        train_set = LungRadiomicsClinicalDataset(
            train_paths,
            path_to_info,
            'train',
            train_transforms,
            patch_size,
            gt='manual'
            )
        valil_set = LungRadiomicsClinicalDataset(
            valid_paths,
            path_to_info,
            'valid',
            val_transforms,
            patch_size,
            gt='manual'
            )
        test_set = LungRadiomicsClinicalDataset(
            test_paths,
            path_to_info,
            'test',
            val_transforms,
            patch_size,
            gt='manual'
            )
    elif args.feats == 'cnn':
        train_set = LungRadiomicsClinicalDataset(
            train_paths,
            path_to_info,
            'train',
            train_transforms,
            patch_size,
            gt='manual'
            )
        valil_set = LungRadiomicsClinicalDataset(
            valid_paths,
            path_to_info,
            'valid',
            val_transforms,
            patch_size,
            gt='manual'
            )
        test_set = LungRadiomicsClinicalDataset(
            test_paths,
            path_to_info,
            'test',
            val_transforms,
            patch_size,
            gt='manual'
            )
    
    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=ImbalancedDatasetSampler(train_set), drop_last=True)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valil_set, batch_size=6, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_set , batch_size=6, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model, trainer and pl_module
    if args.feats == 'r':
        model = {
            'E':  MyEncoder(2,args.n_filters,2),
            'classifer': MyClassifer(args.n_filters, 10, 2)
        }
    elif args.feats == 'c':
        model = {
            'E':  MyEncoder(2,args.n_filters,2),
            'classifer': MyClassifer(args.n_filters, 14, 2)
        }
    elif args.feats == 'cr':
        model = {
            'E':  MyEncoder(2,args.n_filters,2),
            'classifer': MyClassifer(args.n_filters, 24, 2)
        }  
    elif args.feats == 'cnn':
        model = {
            'E':  MyEncoder(2,args.n_filters,2),
            'classifer': MyClassifer(args.n_filters, 0, 2)
        }         
    trainer = L.Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'log', tag),
        devices=args.gpus,
        max_epochs=args.epochs,
        accelerator='gpu',
        strategy='auto',
        precision=precision,  
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[
            ModelSummary(),
            LearningRateMonitor('epoch'),
            ModelCheckpoint(monitor='valid_epoch_metric', mode='max', save_top_k=5),
            # StochasticWeightAveraging(swa_lrs=args.learning_rate/100, swa_epoch_start=0.8)
        ]
    )
    # auc_fn = AUROC(task='binary')
    auc_fn = roc_auc_score
    acc_fn = accuracy
    pl_kwargs = {
        'model':        model,
        'batch_size':   args.batch_size,
        'lr':           args.learning_rate,
        'optimizer':    torch.optim.Adam,
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'loss_fn':      CrossEntropyFocalLoss(alpha=args.alpha),
        'metric_fn':    {
            'metric_fn_1':  auc_fn,
            'metric_fn_2':  acc_fn},
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
    print(f"validation metric: {valid_result[0]['test_epoch_metric']:.4f}")
    print(f"test metric: {test_result[0]['test_epoch_metric']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The parameters of the training')

    parser.add_argument("-t", "--tag", type=str, required=False, default='Latest',
                        help="Tag of this training")
    parser.add_argument("-g", "--gpus", nargs='+', type=int, required=False, default=[2],
                        help="Which GPUs are using in this training")
    parser.add_argument("-r", "--resume", type=bool, required=False, default=False,
                        help="Whether to continue training the model")
    parser.add_argument("-a", "--autocast", type=bool, required=False, default=False,
                        help="Whether to use autocast (torch.cuda.amp) during training")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=4,
                        help="Batch size of the training")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=200,
                        help="Number of epochs of the training")
    parser.add_argument("-w", "--workers", type=int, required=False, default=4,
                        help="Number of workers of the training")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=0.0001,
                        help="Learning rate of the training")
    parser.add_argument("-f", "--n_filters",  type=int, required=False, default=32,
                        help="Number of filters of the training")
    parser.add_argument("-m", "--mode",  type=str, required=False, default='train',
                        help="Train, Test or Predict")
    parser.add_argument("-s", "--split",  type=str, required=False, default='1',
                        help="which splits")
    parser.add_argument("-fe", "--feats",  type=str, required=False, default='c', choices=['r', 'c', 'cr', 'cnn'],
                        help="which splits")
    parser.add_argument("-al", "--alpha",  type=float, required=False, default=0.5,
                        help="which splits")

    args = parser.parse_args()
    main(args)