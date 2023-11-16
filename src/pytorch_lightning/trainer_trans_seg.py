import torch
import lightning as L

class PLModule(L.LightningModule):
    def __init__(self, model, batch_size, lr, optimizer, lr_scheduler, loss_fn, metric_fn):
        super().__init__()
        self.model_E = model['E']
        self.model_D = model['D']
        self.batch_size=batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metric_fn'])

    def forward(self, x):
        x, ds0, ds1, ds2 = self.model_E(x)
        logit = self.model_D(x, ds0, ds1, ds2)
        prob = torch.sigmoid(logit)
        return prob
    
    def configure_optimizers(self):
        optim_paras = [{'params': net.parameters()} for net in [self.model_E, self.model_D]]
        optimizer = self.optimizer(optim_paras, lr=self.lr, betas=(0.9, 0.99))
        lr_scheduler = self.lr_scheduler(optimizer, T_0=25, eta_min=self.lr/100)
        return [optimizer],  [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        sample, info = batch
        x = sample['input']
        prob = self(x)
        target = sample['target']
        loss = self.loss_fn(prob, target)
        metric = self.metric_fn(prob.detach(), target.detach())
        self.log('train_loss', loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('train_metric', metric, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sample, info = batch
        x = sample['input']
        prob = self(x)
        target = sample['target']
        loss = self.loss_fn(prob, target)
        metric = self.metric_fn(prob.detach(), target.detach())
        self.log('valid_loss', loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('valid_metric', metric, batch_size=self.batch_size, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        sample, info = batch
        x = sample['input']
        prob = self(x)
        target = sample['target']
        metric = self.metric_fn(prob.detach(), target.detach())
        self.log('test_metric', metric, batch_size=1, on_epoch=True)