import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 32, 3, 1, groups=1)
        )
        self.adjust = nn.BatchNorm2d(1)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def xcorr(self, z, x):
        batch_size, _, H, W = x.shape
        x = torch.reshape(x, (1, -1, H, W))
        out = F.conv2d(x, z, groups=batch_size)
        xcorr_out = out.transpose(0,1)
        return xcorr_out

    def init_template(self, z):
        z_feat = self.features(z)
        return torch.cat([z_feat for _ in range(3)], dim=0)

    def forward_corr(self, z_feat, x):
        x_feat = self.features(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        return score

    def forward(self, z, x):
        z_feat = self.features(z)
        x_feat = self.features(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        return score
