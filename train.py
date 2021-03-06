from data import Train
from model import get_model

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


class VTraining(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        self.weight = 5

    def forward(self, image, ):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, target = batch
        pred = self.model(image)['out']
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss[target != 0].sum() * self.weight + loss[target == 0].sum()
        loss = loss / target.numel()
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.00002)
        # return optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        return optimizer

    def train_dataloader(self):
        data = Train()
        # data_test = News('test')
        dataloader = torch.utils.data.DataLoader(data, batch_size=18, shuffle=True, num_workers=8)
        return dataloader

    # def val_dataloader(self):
    #     return build_dataloader(512, 'val')


if __name__ == '__main__':
    # init model
    model = VTraining()

    trainer = pl.Trainer(accelerator="dp", gpus=3, num_sanity_val_steps=0, precision=16,
        enable_checkpointing=True, accumulate_grad_batches=1, sync_batchnorm=True, max_epochs=40)
        # resume_from_checkpoint="lightning_logs/version_5/checkpoints/epoch=4-step=25050.ckpt")
    trainer.fit(model=model)
