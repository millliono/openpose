import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, save_for_loss_pafs, save_for_loss_htmps, paf_target, htmp_target):
        total_loss = 0

        for paf in save_for_loss_pafs:
            total_loss += self.mse(paf, paf_target)

        for htmp in save_for_loss_htmps:
            total_loss += self.mse(htmp, htmp_target)

        return total_loss


if __name__ == "__main__":
    loss = Loss()

    import model

    mdl = model.openpose(in_channels=3)
    inp = torch.randn(1, 3, 224, 224)

    out, save_for_loss_pafs, save_for_loss_htmps = mdl(inp)
    ls = loss(
        save_for_loss_pafs,
        save_for_loss_htmps,
        torch.randn(1, 52, 28, 28),
        torch.randn(1, 26, 28, 28),
    )
    print("total error: ", ls)
