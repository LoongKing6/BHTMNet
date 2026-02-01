import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from models.BHTMNet import BHTMNet

import os.path as osp
from utils import get_channel_info
import torchmetrics



def init_model(args):
    model = None

    if args.model == 'BHTMNet':
        model = BHTMNet(
            num_chan=args.num_chan, num_time=args.num_time, layers_transformer = 1, hidden_graph = 64, num_head = 16, alpha = 0.25, temporal_kernel=args.kernel_length, num_kernel=args.T,
            num_classes=args.num_class, depth=int(args.num_layers - 2), heads=args.AT,
            mlp_dim=args.AT, dim_head=args.AT, dropout=args.dropout)



    return model

class DLModel(pl.LightningModule):
    def __init__(self, config):
        super(DLModel, self).__init__()
        self.save_hyperparameters()
        self.net = init_model(config)
        self.test_step_pred = []
        self.test_step_ground_truth = []
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_class)
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_class, average='macro')
        self.config = config

    def forward(self, x):
        return self.net(x)

    def get_metrics(self, pred, y):
        acc = self.acc(pred, y)
        f1 = self.F1(pred, y)
        return acc, f1

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc, "val_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_step_pred.append(y_hat)
        self.test_step_ground_truth.append(y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"test_loss": loss, "test_acc": acc, "test_f1": f1},
            on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_acc": acc, "test_f1": f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )
        return [optimizer], [scheduler]





