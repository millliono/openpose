import torch
import torch.nn as nn


class PoseLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, save_for_loss_pafs, save_for_loss_htmps, paf_target, htmp_target, mask_out):
        total_loss = 0

        for paf in save_for_loss_pafs:
            # paf = paf * mask_out
            # paf_target = paf_target * mask_out
            total_loss += self.mse(paf, paf_target)

        for htmp in save_for_loss_htmps:
            # htmp = htmp * mask_out
            # htmp_target = htmp_target * mask_out
            total_loss += self.mse(htmp, htmp_target)

        return total_loss
