import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyValueNet(nn.Module):
    def __init__(self,
                 position_dim=8,  # 位置分布维度 (8个棋子)
                 action_space=2700,  # 亚马逊棋动作空间 (10x10棋盘估算)
                 hidden_dim=128,
                 dropout=0.2):
        super(PolicyValueNet, self).__init__()

        self.position_dim = position_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim

        # ============ 编码器部分 ============

        # 1. 当前真实状态编码器
        self.current_encoder = nn.Sequential(
            nn.Linear(position_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 2. TCN预测状态编码器
        self.tcn_encoder = nn.Sequential(
            nn.Linear(position_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 3. 时序偏差编码器
        # 输入: [current_step, tcn_expected_step, position_deviation, temporal_velocity]
        self.temporal_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 4. 位置差异编码器 (处理两个分布的差异)
        self.diff_encoder = nn.Sequential(
            nn.Linear(position_dim, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.ReLU()
        )

        # ============ 简化的相关性计算 ============

        # 使用余弦相似度计算当前状态与TCN预测的相关性
        # 这比多头注意力更简单，但仍能捕捉特征间的关系

        # ============ 融合网络 ============

        # 特征融合 (64 + 64 + 32 + 32 = 192)
        self.fusion_layer = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============ 输出头 ============

        # Policy头 - 输出动作概率分布
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_space),
        )

        # Value头 - 输出状态价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出 [-1, 1] 的价值
        )

        # ============ 时序校正头 ============

        # 输出时序校正信息，帮助理解游戏节奏
        self.temporal_correction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [节奏加速度, 置信度, 预期步数修正]
        )

    def compute_temporal_features(self, current_step, tcn_expected_step,
                                  current_pos, tcn_predicted_pos):
        """计算时序相关特征"""
        batch_size = current_pos.size(0)

        # 1. 位置差异程度 (L2距离) - 确保输出始终是 [batch_size, 1]
        position_diff = current_pos - tcn_predicted_pos  # [batch_size, 8]
        position_deviation = torch.norm(position_diff, dim=-1, keepdim=True)  # [batch_size, 1]

        # 确保position_deviation是二维的 [batch_size, 1]
        if position_deviation.dim() > 2:
            # 如果是 [batch_size, 1, feature] 形状，压缩多余维度
            position_deviation = position_deviation.view(batch_size, 1)
        elif position_deviation.dim() == 1:
            # 如果是 [batch_size] 形状，添加特征维度
            position_deviation = position_deviation.unsqueeze(1)

        # 2. 时间步差异 - 确保是正确的形状
        if current_step.dim() == 0:  # 如果是标量
            current_step = current_step.unsqueeze(0)
        if tcn_expected_step.dim() == 0:  # 如果是标量
            tcn_expected_step = tcn_expected_step.unsqueeze(0)

        # 确保batch维度匹配
        if current_step.size(0) == 1 and batch_size > 1:
            current_step = current_step.expand(batch_size)
        if tcn_expected_step.size(0) == 1 and batch_size > 1:
            tcn_expected_step = tcn_expected_step.expand(batch_size)

        step_diff = (current_step - tcn_expected_step).float().unsqueeze(1)  # [batch_size, 1]

        # 3. 计算"游戏速度" (位置变化/时间变化的比率)
        temporal_velocity = position_deviation / (torch.abs(step_diff) + 1e-6)  # [batch_size, 1]

        # 4. 当前步数的归一化
        normalized_step = current_step.float().unsqueeze(1) / 300.0  # [batch_size, 1]

        # 调试信息 - 确保所有张量形状正确
        print(f"Debug shapes - normalized_step: {normalized_step.shape}, step_diff: {step_diff.shape}, "
              f"position_deviation: {position_deviation.shape}, temporal_velocity: {temporal_velocity.shape}")

        # 确保所有张量都是 [batch_size, 1] 的形状后再拼接
        temporal_features = torch.cat([
            normalized_step,  # [batch_size, 1]
            step_diff,  # [batch_size, 1]
            position_deviation,  # [batch_size, 1]
            temporal_velocity  # [batch_size, 1]
        ], dim=1)  # 结果: [batch_size, 4]

        return temporal_features

    def forward(self, current_state, tcn_predicted_state,
                current_step, tcn_expected_step):
        """
        Args:
            current_state: [batch_size, 8] 当前真实的位置分布
            tcn_predicted_state: [batch_size, 8] TCN预测的位置分布
            current_step: [batch_size] 当前游戏步数
            tcn_expected_step: [batch_size] TCN预期的步数

        Returns:
            policy: [batch_size, action_space] 动作概率分布
            value: [batch_size, 1] 状态价值
            temporal_correction: [batch_size, 3] 时序校正信息
        """
        batch_size = current_state.size(0)

        # ============ 特征编码 ============

        # 1. 编码当前状态和TCN预测状态
        current_features = self.current_encoder(current_state)  # [B, 64]
        tcn_features = self.tcn_encoder(tcn_predicted_state)  # [B, 64]

        # 2. 计算并编码时序特征
        temporal_features = self.compute_temporal_features(
            current_step, tcn_expected_step, current_state, tcn_predicted_state
        )
        temporal_encoded = self.temporal_encoder(temporal_features)  # [B, 32]

        # 3. 编码位置差异
        position_diff = current_state - tcn_predicted_state
        diff_encoded = self.diff_encoder(position_diff)  # [B, 32]

        # ============ 注意力机制 ============

        # 简化的注意力机制，避免复杂的多头注意力问题
        # 计算当前状态和TCN预测的相关性
        similarity = torch.cosine_similarity(current_features, tcn_features, dim=1)  # [B]
        attention_weights = torch.sigmoid(similarity).unsqueeze(1) # [B, 1], 范围 [0, 1]

        # 基于相关性加权融合特征
        attended_features = attention_weights * current_features + (1 - attention_weights) * tcn_features  # [B, 64]

        # ============ 特征融合 ============

        # 融合所有特征: current + attended + temporal + diff
        fused_features = torch.cat([
            current_features,  # 64
            attended_features,  # 64
            temporal_encoded,  # 32
            diff_encoded  # 32
        ], dim=1)  # [B, 192]

        hidden = self.fusion_layer(fused_features)  # [B, hidden_dim]

        # ============ 输出计算 ============

        # Policy输出
        policy = self.policy_head(hidden)  # [B, action_space]

        # Value输出
        value = self.value_head(hidden)  # [B, 1]

        # 时序校正输出
        temporal_correction = self.temporal_correction_head(hidden)  # [B, 3]

        return policy, value, temporal_correction, attention_weights


# ============ TCN集成包装器 ============

class TCNPolicyValueIntegrator(nn.Module):
    """将TCN与Policy-Value网络集成的包装器"""

    def __init__(self, tcn_model, policy_value_net):
        super(TCNPolicyValueIntegrator, self).__init__()
        self.tcn = tcn_model
        self.policy_value = policy_value_net

    def forward(self, input1, input2, num, input_prev=None, input3=None, current_step=None):
        """
        集成TCN和Policy-Value的前向传播

        Args:
            input1, input2, num, input_prev, input3: TCN的输入参数
            current_step: 当前游戏步数

        Returns:
            tcn_output: TCN的预测输出
            policy: 校正后的动作概率
            value: 校正后的状态价值
            temporal_info: 时序校正信息
        """
        # 1. TCN前向传播
        tcn_output = self.tcn(input1, input2, num, input_prev, input3)

        # 2. 如果没有提供当前步数，跳过Policy-Value
        if current_step is None:
            return tcn_output, None, None, None

        # 3. 准备Policy-Value的输入
        # 假设input1是当前状态的位置分布
        current_state = input1 if input1.dim() == 2 else input1.mean(dim=0).unsqueeze(0)
        tcn_predicted_state = tcn_output if tcn_output.dim() == 2 else tcn_output.unsqueeze(0)

        # TCN预期的步数 (假设是当前步数+1)
        tcn_expected_step = current_step + 1

        # 4. Policy-Value前向传播
        policy, value, temporal_correction, attention_weights = self.policy_value(
            current_state=current_state,
            tcn_predicted_state=tcn_predicted_state,
            current_step=current_step,
            tcn_expected_step=tcn_expected_step
        )

        temporal_info = {
            'correction': temporal_correction,
            'attention': attention_weights,
            'deviation': torch.norm(current_state - tcn_predicted_state, dim=1)
        }

        return tcn_output, policy, value, temporal_info


# ============ 训练相关的损失函数 ============

class PolicyValueLoss(nn.Module):
    """Policy-Value网络的联合损失函数"""

    def __init__(self, policy_weight=1.0, value_weight=0.5, temporal_weight=0.2):
        super(PolicyValueLoss, self).__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.temporal_weight = temporal_weight

    def forward(self, policy_logits, value_pred, temporal_pred,
                policy_target_indices, value_target, temporal_target):
        # policy_target_indices: LongTensor of shape [B], 每个元素是 0..C-1
        # policy_logits: [B, C] (raw logits)
        policy_loss = F.cross_entropy(policy_logits, policy_target_indices)

        value_loss = F.mse_loss(value_pred, value_target)
        temporal_loss = F.mse_loss(temporal_pred, temporal_target)

        total_loss = (self.policy_weight * policy_loss +
                      self.value_weight * value_loss +
                      self.temporal_weight * temporal_loss)

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'temporal_loss': temporal_loss.item()
        }


