from data import TrainV2
from model import get_modelv2

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


class V2Training(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_modelv2()
        self.weight = 5

    def forward(self, image, ):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, target, image2, target2, target1_boxes, target2_boxes, target1_reid, target2_reid = batch

        image = torch.cat([image, image2], dim=0)
        target = torch.cat([target, target2], dim=0)
        results = self.model(image)
        feat, pred = results['feat'], results['out']
        feat1, feat2 = feat[:feat.shape[0]//2], feat[feat.shape[0]//2:]
        feat1_vec = []
        for b in range(1,7):
            mask = target1_boxes == b
            # bs, c
            feat1_vec.append((feat1 * mask).sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[2, 3], keepdim=True) + 1e-8))
        # bs, n, c, 1, 1
        feat1_vec = torch.stack(feat1_vec, dim=1)
        feat2_vec = []
        for b in range(1,7):
            mask = target2_boxes == b
            # bs, c
            feat2_vec.append((feat2 * mask).sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[2, 3],keepdim=True) + 1e-8))
        feat2_vec = torch.stack(feat2_vec, dim=1)
        # bs, n, c
        feat_vec = torch.cat([feat2_vec, feat1_vec], dim=0)
        
        re_id = self.model.reid_run(feat, feat_vec)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss[target != 0].sum() * self.weight + loss[target == 0].sum()
        loss = loss / target.numel()
        loss = loss + F.cross_entropy(re_id, torch.cat([target1_reid, target2_reid], dim=0).long(), ignore_index=-1)
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.00002)
        # return optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return optimizer

    def train_dataloader(self):
        data = TrainV2()
        # data_test = News('test')
        dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True, num_workers=8)
        return dataloader

    # def val_dataloader(self):
    #     return build_dataloader(512, 'val')


if __name__ == '__main__':
    # init model
    model = V2Training()

    trainer = pl.Trainer(accelerator="dp", gpus=2, num_sanity_val_steps=0, precision=16, gradient_clip_val=0.5,
        enable_checkpointing=True, accumulate_grad_batches=1, sync_batchnorm=True, max_epochs=20)
        # resume_from_checkpoint="lightning_logs/version_7/checkpoints/epoch=0-step=2669.ckpt")
    trainer.fit(model=model)
