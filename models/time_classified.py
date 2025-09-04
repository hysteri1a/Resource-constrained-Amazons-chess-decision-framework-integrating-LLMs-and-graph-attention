import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TimeClassified(nn.Module):
    def __init__(self, m, n, lowerlimit):  # m是     ;n是
        super(TimeClassified, self).__init__()
        self.n = n
        self.m = m
        assert m <= n
        self.lowerlimit = lowerlimit
        self.convlist = nn.ModuleList(
            [weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1)) for i in range(m - 1)])
        self.relulist = nn.ModuleList([nn.ReLU(inplace=False) for i in range(m - 1)])

        # 全连接层
        self.fc = nn.Linear(n - m + 1, self.lowerlimit)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.lowerlimit, stride=1)  # 使用卷积将输出变为 1x1
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):  # tensor [[],[],[]]
        torch.autograd.set_detect_anomaly(True)
        assert self.n - self.m + 1 >= x.size(1)
        for conv, relu in zip(self.convlist, self.relulist):
            x = conv(x)  # 使用 nn.ModuleList 的卷积层
            x = relu(x)  # 使用 nn.ModuleList 的 ReLU 激活层
            x = self.dropout(x)
            if x.size(1) == 1:
                break
        if self.n - self.m + 1 != self.lowerlimit:
            batch_size = x.size(0)
            if x.size(2) < self.n - self.m + 1:
                x = nn.functional.pad(x, (0, self.n - self.m + 1 - x.size(2)), mode='constant', value=0)
            if x.size(2) > self.n - self.m + 1:
                x = x[:, :, :self.n - self.m + 1]
            x = self.fc(x)
            x = x.view(batch_size, 1, -1)  # 形状变为 (batch_size, 1, 3)
        # 使用卷积层，输出形状将变为 (batch_size, 1, 1)
        x = self.conv(x)
        return x
