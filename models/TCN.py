import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Conv2DWithPadding(nn.Module):
    def __init__(self):
        super(Conv2DWithPadding, self).__init__()
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(2, 3), stride=1, padding=0)
        self.conv2d = weight_norm(self.conv2d)

    def expand_tensor(self, input):
        """
        扩展输入张量
        """
        # 先对 input 进行左侧填充0，右侧填充1（暂时填充0）
        expanded_tensor = F.pad(input, (1, 1, 0, 0), mode='constant', value=0)

        expanded2 = expanded_tensor.clone()
        # 将最右侧那列全部设置为 5
        expanded2[:, -1] = 5

        return expanded2

    def forward(self, input1, input2):
        """
            执行卷积操作mm
            """
        input_tensor = torch.cat((input1, input2), dim=0)
        print(input_tensor)
        if input_tensor.size(0) < 2:
            # 不足以执行 Conv2D，直接返回 input_tensor 或填充一个默认值
            return input_tensor[-1].clone()  # 或者 return torch.zeros_like(input_tensor[-1])

        # 扩展输入张量
        expanded_tensor = self.expand_tensor(input_tensor)
        # 增加 batch 维度和 channel 维度
        x = expanded_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 10)

        out = self.conv2d(x)  # (1,1,*,*)
        # squeeze 回去，同时不做原地
        out = out.squeeze(0).squeeze(0)
        return out


class Conv1DWithPadding(nn.Module):
    def __init__(self):
        super(Conv1DWithPadding, self).__init__()
        # 使用 1x3 的卷积核
        self.conv2d = weight_norm(nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=0))

    def expand_tensor(self, input):
        """
        扩展输入张量
        """
        # 对 input 进行左侧填充0，右侧填充5
        padded = F.pad(input, (1, 1, 0, 0), mode='constant', value=0)
        padded2 = padded.clone()
        padded2[:, -1] = 5
        return padded2

    def forward(self, input1):
        # pad → clone → conv
        expanded = self.expand_tensor(input1)  # shape (N, L+2)
        x = expanded.unsqueeze(0)  # (1,1,L+2)
        out = self.conv2d(x)  # (1,1,*,*)
        # 只在输出上做 squeeze，不影响 autograd
        return out.squeeze(0)


class flexiblelayers(nn.Module):
    def __init__(self, maxlimit, dropout=0.3):
        super(flexiblelayers, self).__init__()
        self.maxlimit = maxlimit
        self.convlist1 = nn.ModuleList([Conv2DWithPadding() for i in range(maxlimit)])
        self.convlist2 = nn.ModuleList([Conv1DWithPadding() for i in range(maxlimit)])
        self.relulist1 = nn.ModuleList([nn.ReLU() for i in range(maxlimit)])
        self.relulist2 = nn.ModuleList([nn.ReLU() for i in range(maxlimit)])
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input1, input2, num):
        num = min(num, self.maxlimit - 1)
        out = self.dropout1(self.relulist1[num](self.convlist1[num](input1, input2)))
        out = self.dropout2(self.relulist2[num](self.convlist2[num](out)))
        return self.relu(out + input1)


class flexibleTCN(nn.Module):
    def __init__(self, maxlimit, model):  # maxlimit是为了截断过长的序列
        super(flexibleTCN, self).__init__()
        self.layers = flexiblelayers(maxlimit)
        self.weight = model  # timeclassified（自回归子模块）

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, num: int,
                input_prev: torch.Tensor = None, input3: torch.Tensor = None) -> torch.Tensor:
        """
        input1, input2: TCN 层输入
        num: 当前层索引
        input_prev: 自回归模型输入
        input3: 上一层预测值，用于残差连接
        返回: 张量形状 [batch_size, features]
        """
        # 1) TCN 前向
        tcn_output = self.layers(input1, input2, num)
        # 如果输出为 [B,1,D]，压缩成 [B,D]
        if tcn_output.dim() == 3 and tcn_output.size(1) == 1:
            tcn_output = tcn_output.squeeze(1)

        # 2) 若无残差分支，直接返回
        if input3 is None or input_prev is None:
            return tcn_output

        # 3) 自回归残差分支
        res = input3
        # 同样确保 res 为 [B,D]
        if res.dim() == 3 and res.size(1) == 1:
            res = res.squeeze(1)

        # ar_output: [B,D]
        ar_output = self.weight(input_prev)

        # 残差融合：tcn + (res - tcn) * ar_output
        output = tcn_output + (res - tcn_output) * ar_output
        return output