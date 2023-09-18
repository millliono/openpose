import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, save_for_loss_pafs, save_for_loss_htmps, paf_target, htmp_target, mask_out):
        total_loss = 0

        for paf in save_for_loss_pafs:
            paf = paf * mask_out
            paf_target = paf_target * mask_out
            total_loss += self.mse(paf, paf_target)

        for htmp in save_for_loss_htmps:
            htmp = htmp * mask_out
            htmp_target = htmp_target * mask_out
            total_loss += self.mse(htmp, htmp_target)

        return total_loss


if __name__ == "__main__":
    import model

    loss = PoseLoss()
    model = model.openpose(in_channels=3)
    inp = torch.randn(1, 3, 224, 224)

    out, save_for_loss_pafs, save_for_loss_htmps = model(inp)
    ls = loss(
        save_for_loss_pafs,
        save_for_loss_htmps,
        torch.randn(1, 52, 28, 28),
        torch.randn(1, 26, 28, 28),
    )
    print("total error: ", ls)
