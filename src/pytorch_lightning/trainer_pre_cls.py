import os
import torch
import pickle
import lightning as L

class PLModule(L.LightningModule):
    def __init__(self, model, batch_size, lr, optimizer, lr_scheduler, loss_fn, metric_fn, alpha=0.75):
        super().__init__()
        self.model_E = model['E']
        self.model_classifer = model['classifer']
        self.m = torch.nn.Softmax(dim=1)
        self.batch_size=batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.metric_fn_1 = metric_fn['metric_fn_1']
        self.metric_fn_2 = metric_fn['metric_fn_2']
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metric_fn_1', 'metric_fn_2'])
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x, radiomics_feats):
        x, ds0, ds1, ds2 = self.model_E(x)
        class_logit = self.model_classifer(x, radiomics_feats)
        class_prob = self.m(class_logit)
        return class_prob, x, ds0, ds1, ds2
    
    def configure_optimizers(self):
        optim_paras = [{'params': net.parameters()} for net in [self.model_E, self.model_classifer]]
        optimizer = self.optimizer(optim_paras, lr=self.lr, betas=(0.9, 0.99))
        lr_scheduler = self.lr_scheduler(optimizer, T_0=25, eta_min=self.lr/100)
        return [optimizer],  [lr_scheduler]        
    
    def training_step(self, batch, batch_idx):
        sample, info = batch
        x, radiomics_feats = sample['input'], sample['feature']
        class_prob, x, ds0, ds1, ds2 = self(x, radiomics_feats)
        class_label = sample['class_label'].type(torch.cuda.LongTensor)
        loss = self.loss_fn(class_prob, class_label)
        batch_acc = self.metric_fn_2(class_prob.detach(), class_label.detach())
        self.log('train_loss', loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('train_acc', batch_acc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sample, info = batch
        x, radiomics_feats = sample['input'], sample['feature']
        class_prob, x, ds0, ds1, ds2 = self(x, radiomics_feats)
        class_label = sample['class_label'].type(torch.cuda.LongTensor)
        self.validation_step_outputs.append(class_prob)
        self.validation_step_targets.append(class_label)
        loss = self.loss_fn(class_prob, class_label)
        batch_acc = self.metric_fn_2(class_prob.detach(), class_label.detach())
        self.log('valid_loss', loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('valid_acc', batch_acc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs)
        all_targets = torch.stack(self.validation_step_targets)
        auc = self.metric_fn_1(all_targets.cpu().numpy(), all_preds.cpu().numpy()[:,1])
        acc = self.metric_fn_2(all_preds, all_targets)
        epoch_metric = self.alpha * auc + (1-self.alpha) * acc
        self.log('valid_epoch_metric', epoch_metric)
        self.log('valid_epoch_auc', auc)
        self.log('valid_epoch_acc', acc)
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        sample, info = batch
        x, radiomics_feats = sample['input'], sample['feature']
        class_prob, x, ds0, ds1, ds2 = self(x, radiomics_feats)
        class_label = sample['class_label'].type(torch.cuda.LongTensor)
        self.test_step_outputs.append(class_prob)
        self.test_step_targets.append(class_label)
        batch_acc = self.metric_fn_2(class_prob.detach(), class_label.detach())
        self.log('test_acc', batch_acc, batch_size=1, on_epoch=True)

        
    def on_test_epoch_end(self) -> None:
        all_preds = torch.stack(self.test_step_outputs)
        all_targets = torch.stack(self.test_step_targets)
        auc = self.metric_fn_1(all_targets.cpu().numpy(), all_preds.cpu().numpy()[:,1])
        acc = self.metric_fn_2(all_preds, all_targets)
        epoch_metric = self.alpha * auc + (1-self.alpha) * acc
        self.log('test_epoch_metric', epoch_metric)
        self.log('test_epoch_auc', auc)
        self.log('test_epoch_acc', acc)
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()