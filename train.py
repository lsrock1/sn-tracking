from data import Train
from model import get_model

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


def main():
    model = get_model()
    epochs = 100
    batch_size = 8
    lr = 0.001
    wd = 0.00002
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    data = Train()
    # data_test = News('test')
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    # dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8)

    for epoch in range(epochs):
        loss_ = 0.
        for idx, (names, target) in enumerate(dataloader):
            names = names.cuda()
            target = target.cuda()
            model.zero_grad()
            news = model(names)['out']
            loss = F.mse_loss(news, target)
            # loss = F.mse_loss(news, target, reduction='none')[target != 0]
            loss = loss.mean()# + F.cross_entropy(bjd, names['bjd_code'])
            loss.backward()
            optimizer.step()
            loss_ += loss.item()
            # accuracy = (news.detach().argmax(1) == target).float().mean()
            print(f'{epoch}, {idx+1}/{len(dataloader)}: ', loss_/(idx+1))
        scheduler.step()

        # total = 0
        # correct = 0
        # for idx, (names, target) in enumerate(dataloader_test):
        #     names = {key: value.cuda() for key, value in names.items()}
        #     target = target.cuda()
        #     news, bjd = model(names)
        #     # loss = F.cross_entropy(news, target)
        #     # loss_ += loss.item()
        #     # accuracy = (news.detach().argmax(1) == target).float().mean()
        #     total += len(names['input_ids'])
        #     correct += (news.detach().argmax(1) == target).float().sum().item()
        # print(f'{epoch}, {correct/total}')
        torch.save(model.state_dict(), f'model{epoch}.pt')


class VTraining(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, image, ):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, target = batch
        pred = self.model(image)['out']
        loss = F.mse_loss(pred, target)
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
