import torch.nn as nn


class PoseLoss(nn.Module):

    def __init__(self, batch_size, reduction):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.batch_size = batch_size

    def forward(self, save_for_loss_pafs, save_for_loss_htmps, paf_target, htmp_target):
        total_loss = 0

        for x in save_for_loss_pafs:
            # paf = paf * mask_out
            # paf_target = paf_target * mask_out
            total_loss += self.mse(x, paf_target)

        for x in save_for_loss_htmps:
            # htmp = htmp * mask_out
            # htmp_target = htmp_target * mask_out
            total_loss += self.mse(x, htmp_target)

        return total_loss / self.batch_size
