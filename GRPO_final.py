import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy
import random
import os
from datetime import datetime

class GRPOBinaryDecisionMaker(nn.Module):
    """
    基于GRPO的二选一决策系统
    核心思想：不是绝对评估两个候选节点，而是基于历史群体决策的相对表现来学习最优选择策略
    """

    def __init__(self, position_dim=8, history_size=1000, learning_rate=0.001):
        super(GRPOBinaryDecisionMaker, self).__init__()

        self.position_dim = position_dim
        self.history_size = history_size

        # 历史决策记录 - 用于群体相对学习
        # 现在我们在记录历史时保存 node 特征 tensor（CPU），以便后续稳定重放
        self.decision_history = deque(
            maxlen=history_size)  # 存储 (node1_features_tensor, node2_features_tensor, choice, outcome)
        self.performance_buffer = deque(maxlen=history_size)  # 存储决策的后续表现

        # 候选节点特征提取器
        self.node_encoder = nn.Sequential(
            nn.Linear(position_dim + 4, 64),  # position + [obj, depth, visits, win_rate, time_step]
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 双节点对比分析器
        self.comparison_network = nn.Sequential(
            nn.Linear(64 + 16, 64),  # 双节点特征(64) + 对比特征(16)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出相对优势分数
        )

        # 群体上下文编码器 - 考虑历史群体决策模式
        self.context_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 群体上下文特征
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # 时序趋势分析器 - 基于TCN预测调整决策权重
        self.temporal_analyzer = nn.Sequential(
            nn.Linear(position_dim * 5, 24),  # current + tcn_prediction + temporal_features
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.ReLU()
        )

        # 最终决策融合层
        # 我们使用 comparison_score (1) + context(16) + temporal(8) = 25
        self.decision_fusion = nn.Sequential(
            nn.Linear(1 + 16 + 8, 32),  # comparison_score (1) + context(16) + temporal(8) = 25
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        # 性能跟踪
        self.recent_decisions = []
        self.performance_stats = {
            'total_decisions': 0,
            'correct_predictions': 0,
            'average_confidence': 0.0,
            'decision_diversity': 0.0
        }
        self.ensure_model_loaded(model_dir='./train', model_filename='grpo_final_model.pth', auto_create=True)

        self.loss_log_path = os.path.join('train', 'loss_log.txt')

    def _append_loss_log(self, loss_value, info=None, filepath=None, include_date=True):
        """
        将训练结果（loss）追加到文本文件，每次一行。
        格式（默认含日期，日期在行尾，用空格分割）:
           <loss> <info (optional)> <ISO-datetime>
        如果 include_date=False，则为:
           <loss> <info (optional)>

        例如:
           0.034512 valid_samples=28,lr=0.001 2025-08-21T11:23:45
           0.029874 valid_samples=30,lr=0.001 2025-08-21T11:25:12

        note: 使用空格分割，便于后续用 awk/split 等工具处理。
        """
        try:
            if filepath is None:
                filepath = self.loss_log_path
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 准备 loss 字符串（保留 6 位小数）
            if isinstance(loss_value, float):
                loss_str = f"{loss_value:.6f}"
            else:
                # 仍尝试把可转为 float 的值格式化，否则直接 str()
                try:
                    loss_str = f"{float(loss_value):.6f}"
                except Exception:
                    loss_str = str(loss_value)

            parts = [loss_str]

            if info:
                # 把 info 中的逗号或制表符合并为无空格项，方便按字段 split（可按需调整）
                info_str = str(info)
                parts.append(info_str)

            if include_date:
                parts.append(datetime.now().isoformat(timespec='seconds'))

            line = " ".join(parts)

            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except Exception as e:
            # 日志写入失败不影响训练流程，但打印提示以便调试
            print(f"⚠ Failed to append loss log: {e}")

    def model_path(self, model_dir='./train', model_filename='grpo_final_model.pth'):
        """返回拼好的模型文件路径（不做 I/O）"""
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, model_filename)

    def ensure_model_loaded(self, model_dir='./train', model_filename='grpo_final_model.pth', auto_create=True,
                            device=None):
        """
        尝试从 model_dir/model_filename 加载模型（包括历史等）。
        - 如果文件存在：调用 self.load_model(path)
        - 如果文件不存在且 auto_create=True：保存当前（fresh）模型到该路径（调用 self.save_model(path)）
        - 将模型移动到指定 device（若未指定，则自动选择 cuda/ cpu）
        返回 True 表示加载或创建成功，False 表示失败。
        """
        try:
            path = self.model_path(model_dir, model_filename)

            # 选择设备
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if os.path.exists(path):
                try:
                    # load_model 已在类中定义（负责恢复 history 等）
                    self.load_model(path)
                    # 把 model 移到目标设备
                    self.to(device)
                    print(f"✓ GRPO model loaded from: {path} -> device: {device}")
                    return True
                except Exception as e:
                    print(f"⚠ Failed to load model from {path}: {e}")
                    # 继续到创建/保存初始模型（如果允许）
            # 文件不存在 或 加载失败
            if auto_create:
                try:
                    # save_model 已在类中定义（会保存 state_dict + history）
                    self.save_model(path)
                    # 将模型移动到 device
                    self.to(device)
                    print(f"✓ No existing model found. Saved initial model to: {path}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to save initial model to {path}: {e}")
                    return False
            else:
                print(f"ℹ Model file not found at {path} and auto_create is False.")
                return False

        except Exception as e:
            print(f"Unexpected error in ensure_model_loaded: {e}")
            return False

    # -------------------- 特征 / 辅助 --------------------
    def _model_device(self):
        return next(self.parameters()).device

    def extract_node_features(self, node, time_step=0):
        """
        projection: 如果已知，直接传入长度为 position_dim 的 iterable 或 torch tensor。
        如果为 None，尝试使用 node.projection（如果存在）。
        返回 torch.tensor 长度 = position_dim + 5（在 model 的 device 上）
        """
        try:
            device = self._model_device()
            # 确定 position features 源
            proj = node.projection

            print(proj)
            position_features = torch.tensor(proj, dtype=torch.float32, device=device)

            obj_score = float(getattr(node, 'obj', 0.0))
            depth = float(getattr(node, 'depth', 0))  # 要算
            visits = float(len(getattr(node, 'children', [])))

            tail = torch.tensor([obj_score, depth, visits, float(time_step)], dtype=torch.float32, device=device)
            node_features = torch.cat([position_features.float(), tail])

            return node_features
        except Exception as e:
            device = self._model_device()
            import traceback
            traceback.print_exc()
            print(f"Error extracting node features (simplified): {e}")
            return torch.zeros(self.position_dim + 5, device=device)

    def _extract_position_from_board(self, board):
        """从棋盘状态提取位置分布特征（保留作后备）"""
        try:
            if hasattr(board, 'board') and callable(board.board):
                board_array = np.array(board.board())
            elif hasattr(board, '__call__'):
                board_array = np.array(board())
            else:
                board_array = np.array(board) if board is not None else np.zeros((10, 10))

            position_stats = []
            for piece_type in range(1, 5):  # 假设棋子类型1-4
                positions = np.where(board_array == piece_type)
                if len(positions[0]) > 0:
                    center_x = np.mean(positions[0]) / 9.0
                    center_y = np.mean(positions[1]) / 9.0
                else:
                    center_x, center_y = 0.5, 0.5
                position_stats.extend([center_x, center_y])

            # 返回在 model device 上
            device = self._model_device()
            return torch.tensor(position_stats[:self.position_dim], dtype=torch.float32, device=device)

        except Exception as e:
            print(f"Error extracting position from board: {e}")
            return torch.zeros(self.position_dim, device=self._model_device())

    def compute_comparison_features(self, node1_features, node2_features):
        """
        确保输出16维。node?_features 长度 = position_dim + 5 (13)
        """
        device = node1_features.device
        dtype = node1_features.dtype

        feature_diff = torch.abs(node1_features - node2_features)
        feature_ratio = node1_features / (node2_features + 1e-8)

        cosine_sim = F.cosine_similarity(node1_features.unsqueeze(0), node2_features.unsqueeze(0), dim=1)
        euclidean_dist = torch.norm(node1_features - node2_features)

        obj1, obj2 = node1_features[-5], node2_features[-5]
        obj_advantage = (obj1 > obj2).float()
        depth_advantage = (node1_features[-4] > node2_features[-4]).float()
        visits_advantage = (node1_features[-3] > node2_features[-3]).float()

        parts = [
            feature_diff[:3],  # 3
            feature_ratio[-2:],  # 2 -> 5
            cosine_sim.to(dtype),  # 1 -> 6
            euclidean_dist.unsqueeze(0).to(dtype),  # 1 -> 7
            obj_advantage.unsqueeze(0),  # 1 -> 8
            depth_advantage.unsqueeze(0),  # 1 -> 9
            visits_advantage.unsqueeze(0),  # 1 ->10
            (obj1 + obj2).unsqueeze(0),  # 1 ->11
            torch.abs(obj1 - obj2).unsqueeze(0)  # 1 ->12
        ]
        print(parts)
        # 将所有 parts 移到 device
        parts = [p.to(device) if isinstance(p, torch.Tensor) and p.device != device else p for p in parts]
        x = torch.cat(parts)

        # 补齐到16维（pad 0）
        if x.numel() < 16:
            pad = torch.zeros(16 - x.numel(), device=device, dtype=dtype)
            x = torch.cat([x, pad])

        return x

    def compute_group_context(self):
        """计算群体决策上下文 - 基于历史决策模式"""

        if len(self.decision_history) < 10:
            return torch.zeros(10, device=self._model_device())

        recent_decisions = list(self.decision_history)[-50:]

        # 1. 决策一致性 - 相似情况下的选择模式
        consistency_score = self._compute_decision_consistency(recent_decisions)

        # 2. 性能趋势 - 最近决策的成功率趋势
        performance_trend = self._compute_performance_trend()

        # 3. 选择偏好 - 对不同类型节点的偏好模式
        selection_bias = self._compute_selection_bias(recent_decisions)

        # 4. 决策置信度分布
        confidence_stats = self._compute_confidence_stats()

        device = self._model_device()
        context_features = torch.tensor([
            consistency_score,
            performance_trend,
            selection_bias,
            confidence_stats,
            len(self.decision_history) / self.history_size,  # 历史丰富程度
            self.performance_stats['correct_predictions'] / max(1, self.performance_stats['total_decisions']),
            self.performance_stats['average_confidence'],
            self.performance_stats['decision_diversity'],
            self._compute_recent_success_rate(),
            self._compute_decision_volatility()
        ], dtype=torch.float32, device=device)

        return context_features

    def _compute_decision_consistency(self, decisions):
        """计算决策一致性分数"""
        if len(decisions) < 5:
            return 0.5

        consistent_patterns = 0
        total_comparisons = 0

        for i in range(len(decisions) - 1):
            for j in range(i + 1, min(i + 10, len(decisions))):  # 比较邻近的决策
                try:
                    dec1 = decisions[i]
                    dec2 = decisions[j]

                    # dec? 现在可能存的是 feature tensors 或已序列化的结构
                    # 期望格式: (node1_features_tensor or node1_obj-like, node2_features, choice, outcome)
                    def obj_from_entry(d):
                        n1 = d[0]
                        n2 = d[1]
                        # 若为 tensor，取对应位置上的 obj（索引 -5）
                        if isinstance(n1, torch.Tensor) and isinstance(n2, torch.Tensor):
                            return float(n1[-5].item() - n2[-5].item())
                        # 否则尝试访问属性
                        try:
                            return float(getattr(n1, 'obj', 0) - getattr(n2, 'obj', 0))
                        except:
                            return 0

                    obj_diff1 = obj_from_entry(dec1)
                    obj_diff2 = obj_from_entry(dec2)

                    if abs(obj_diff1) < 0.1 and abs(obj_diff2) < 0.1:
                        if dec1[2] == dec2[2]:
                            consistent_patterns += 1
                        total_comparisons += 1
                except Exception:
                    continue

        return consistent_patterns / max(1, total_comparisons)

    def _compute_performance_trend(self):
        """计算性能趋势"""
        if len(self.performance_buffer) < 10:
            return 0.0

        recent_performance = list(self.performance_buffer)[-20:]
        if len(recent_performance) < 10:
            return 0.0

        weights = np.arange(len(recent_performance))
        try:
            trend = np.polyfit(weights, recent_performance, 1)[0]
            return float(trend)
        except Exception:
            return 0.0

    def _compute_selection_bias(self, decisions):
        """计算选择偏好"""
        if len(decisions) < 5:
            return 0.5

        choice_0_count = sum(1 for d in decisions if d[2] == 0)
        return choice_0_count / len(decisions)

    def _compute_confidence_stats(self):
        """计算决策置信度统计"""
        if len(self.recent_decisions) < 5:
            return 0.5

        confidences = [abs(d - 0.5) * 2 for d in self.recent_decisions[-20:]]  # 转换为置信度
        return float(np.mean(confidences))

    def _compute_recent_success_rate(self):
        """计算最近的成功率"""
        if len(self.performance_buffer) < 5:
            return 0.5

        recent_performance = list(self.performance_buffer)[-10:]
        return float(np.mean(recent_performance))

    def _compute_decision_volatility(self):
        """计算决策波动性"""
        if len(self.recent_decisions) < 5:
            return 0.5

        recent_probs = self.recent_decisions[-10:]
        return float(np.std(recent_probs))

    def compute_temporal_features(self, current_projection, tcn_projection=None, time_step=0):
        """
        current_projection, tcn_projection: iterable or torch tensor length position_dim
        返回 length = position_dim * 2 + 3
        """
        try:
            device = self._model_device()
            if not isinstance(current_projection, torch.Tensor):
                cur = torch.tensor(current_projection, dtype=torch.float32, device=device)
            else:
                cur = current_projection.to(device)

            if tcn_projection is None:
                tcn = torch.zeros(self.position_dim, device=device)
            else:
                tcn = torch.tensor(tcn_projection, dtype=torch.float32, device=device) if not isinstance(tcn_projection,
                                                                                                         torch.Tensor) else tcn_projection.to(
                    device)
                if tcn.numel() < self.position_dim:
                    tcn = F.pad(tcn, (0, self.position_dim - tcn.numel())).to(device)
                elif tcn.numel() > self.position_dim:
                    tcn = tcn[:self.position_dim].to(device)

            cur = cur.view(-1, 1)  # -> (8,1) （如果已经是 (8,1) 也安全）
            tcn = tcn.view(-1, 1)  # -> (8,1)

            # temporal_consistency = F.cosine_similarity(cur.unsqueeze(0), tcn.unsqueeze(0), dim=1)[0]
            temporal_consistency = F.cosine_similarity(cur, tcn, dim=1, eps=1e-8).unsqueeze(1)

            # position_deviation = torch.norm(cur - tcn)
            position_deviation = torch.norm(cur - tcn, p=2, dim=1, keepdim=True)

            tp_scalar = torch.tensor(min(1.0, float(time_step) / 100.0), dtype=torch.float32,
                                     device=cur.device)  # scalar on same device
            time_pressure = tp_scalar.view(1, 1).expand(cur.size(0), 1)

            temporal_features = torch.cat([cur, tcn, temporal_consistency, position_deviation, time_pressure], dim=1)

            print(temporal_features.shape)  # should be torch.Size([8, 5])
            return temporal_features.contiguous().view(-1)
        except Exception as e:
            device = self._model_device()
            import traceback
            traceback.print_exc()
            print(f"Error computing temporal features (simplified): {e}")
            return torch.zeros(self.position_dim * 2 + 3, device=device)

    # -------------------- 决策 / 学习 --------------------
    def make_decision(self, node1, node2, current_board_state, tcn_prediction=None, time_step=0):
        """
        GRPO核心决策函数

        Args:
            node1, node2: 两个候选节点
            current_board_state: 当前棋盘状态（这里我们期望是 projection 或 node.projection）
            tcn_prediction: TCN的预测结果
            time_step: 当前时间步

        Returns:
            selected_node: 选择的节点
            decision_confidence: 决策置信度
            decision_info: 详细的决策信息
        """

        self.eval()  # 切换到评估模式
        try:
            with torch.no_grad():
                # 1. 提取节点特征（在 model device 上）
                node1_features = self.extract_node_features(node1, time_step)
                node2_features = self.extract_node_features(node2, time_step)

                # 2. 编码节点特征
                encoded_node1 = self.node_encoder(node1_features)
                encoded_node2 = self.node_encoder(node2_features)

                # 3. 计算对比特征
                comparison_features = self.compute_comparison_features(node1_features, node2_features)
                combined_nodes = torch.cat([encoded_node1, encoded_node2])
                comparison_input = torch.cat([combined_nodes, comparison_features])
                comparison_score = self.comparison_network(comparison_input)

                # 4. 计算群体上下文
                context_features = self.compute_group_context()
                context_encoding = self.context_encoder(context_features)

                # 5. 计算时序特征
                temporal_features = self.compute_temporal_features(current_board_state, tcn_prediction, time_step)
                temporal_encoding = self.temporal_analyzer(temporal_features)

                # 6. 融合所有特征做最终决策
                fusion_input = torch.cat([comparison_score.flatten(), context_encoding, temporal_encoding])
                decision_prob = self.decision_fusion(fusion_input)

                # 7. 做出选择
                choice_idx = 0 if decision_prob.item() > 0.5 else 1
                selected_node = node1 if choice_idx == 0 else node2
                confidence = abs(decision_prob.item() - 0.5) * 2  # 转换为[0,1]的置信度

                # 8. 记录决策历史 — 存 node 特征到 CPU，便于后续 stable replay
                try:
                    self.decision_history.append(
                        (node1_features.detach().cpu(), node2_features.detach().cpu(), choice_idx, None))
                except Exception:
                    # 如果不能保存 tensor（极少见），回退到保存简单信息
                    self.decision_history.append((None, None, choice_idx, None))

                self.recent_decisions.append(decision_prob.item())

                # 9. 更新统计信息
                self.performance_stats['total_decisions'] += 1
                self.performance_stats['average_confidence'] = (
                        self.performance_stats['average_confidence'] * 0.9 + confidence * 0.1
                )

                # 计算决策多样性
                if len(self.recent_decisions) >= 10:
                    recent_choices = [1 if p > 0.5 else 0 for p in self.recent_decisions[-10:]]
                    diversity = 1 - abs(sum(recent_choices) / len(recent_choices) - 0.5) * 2
                    self.performance_stats['decision_diversity'] = diversity

                decision_info = {
                    'decision_prob': decision_prob.item(),
                    'confidence': confidence,
                    'comparison_score': comparison_score.item(),
                    'context_quality': float(torch.mean(context_features).item()),
                    'temporal_consistency': float(temporal_features[-3].item() if len(temporal_features) > 2 else 0.0),
                    'node1_obj': getattr(node1, 'obj', 0.0),
                    'node2_obj': getattr(node2, 'obj', 0.0),
                    'choice_reason': self._generate_choice_reason(decision_prob.item(), comparison_score.item(),
                                                                  confidence)
                }

                return selected_node, confidence, decision_info
        except:
            import traceback
            traceback.print_exc()

    def _generate_choice_reason(self, decision_prob, comparison_score, confidence):
        """生成选择原因的可解释性描述"""
        reasons = []

        if confidence > 0.8:
            reasons.append("高置信度决策")
        elif confidence > 0.6:
            reasons.append("中等置信度决策")
        else:
            reasons.append("低置信度决策")

        if abs(comparison_score) > 0.5:
            reasons.append("对比特征显著差异")

        if decision_prob > 0.7:
            reasons.append("强烈倾向选择节点1")
        elif decision_prob < 0.3:
            reasons.append("强烈倾向选择节点2")
        else:
            reasons.append("选择相对均衡")

        return "; ".join(reasons)

    def update_decision_outcome(self, outcome_score):
        """
        更新最近决策的结果 - 用于后续学习

        Args:
            outcome_score: 决策结果分数 (0-1, 1表示好结果)
        """

        # 更新最近决策的结果
        if self.decision_history:
            # deque 支持索引赋值
            last = self.decision_history[-1]
            # last 可能是 (tensor, tensor, choice, None)
            self.decision_history[-1] = (last[0], last[1], last[2], outcome_score)

        # 添加到性能缓冲区
        self.performance_buffer.append(outcome_score)

        # 更新性能统计
        if outcome_score > 0.6:  # 认为是正确的预测
            self.performance_stats['correct_predictions'] += 1

    def learn_from_batch(self, batch_size=32):
        """
        从历史决策中学习 - GRPO的核心学习机制
        """

        if len(self.decision_history) < batch_size:
            return None

        self.train()  # 切换到训练模式

        # 从历史中采样一个batch
        batch_indices = random.sample(range(len(self.decision_history)), batch_size)
        batch_decisions = [self.decision_history[i] for i in batch_indices]

        total_loss = 0.0
        valid_samples = 0

        for item in batch_decisions:
            # 支持 tuple 为 (node1_feats_tensor, node2_feats_tensor, choice, outcome)
            try:
                node1, node2, choice, outcome = item
            except Exception:
                continue

            if outcome is None:
                continue

            try:
                device = self._model_device()

                # 如果历史中保存的是特征 tensor（CPU），把它搬到 model device
                if isinstance(node1, torch.Tensor) and isinstance(node2, torch.Tensor):
                    node1_features = node1.to(device)
                    node2_features = node2.to(device)
                else:
                    # 退回到从 node 对象重建特征（兼容旧格式）
                    node1_features = self.extract_node_features(node1, 0)
                    node2_features = self.extract_node_features(node2, 0)

                encoded_node1 = self.node_encoder(node1_features)
                encoded_node2 = self.node_encoder(node2_features)

                comparison_features = self.compute_comparison_features(node1_features, node2_features)
                combined_nodes = torch.cat([encoded_node1, encoded_node2])
                comparison_input = torch.cat([combined_nodes, comparison_features])
                comparison_score = self.comparison_network(comparison_input)

                context_features = self.compute_group_context()
                context_encoding = self.context_encoder(context_features)

                temporal_features = torch.zeros(self.position_dim * 2 + 3, device=device)  # 简化的时序特征
                temporal_encoding = self.temporal_analyzer(temporal_features)

                fusion_input = torch.cat([comparison_score.flatten(), context_encoding, temporal_encoding])
                decision_prob = self.decision_fusion(fusion_input)

                # 基于outcome调整目标
                if outcome > 0.6:  # 好结果
                    tgt_val = 1.0 if choice == 0 else 0.0
                else:  # 差结果
                    tgt_val = 0.0 if choice == 0 else 1.0

                target = torch.tensor([tgt_val], dtype=decision_prob.dtype, device=decision_prob.device)

                loss = self.criterion(decision_prob, target)
                total_loss += loss.item()
                valid_samples += 1

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

            except Exception as e:
                print(f"Error in learning step: {e}")
                continue

        average_loss = total_loss / max(1, valid_samples) if valid_samples > 0 else float('inf')

        # ========= 在每次学习后把 loss 写入文件（每行一条） =========
        try:
            # 记录额外信息，便于分析：valid_samples, lr
            lr = self.optimizer.param_groups[0].get('lr', None)
            info = f"valid_samples={valid_samples}, lr={lr}"
            self._append_loss_log(average_loss, info=info)
        except Exception as e:
            print(f"⚠ Failed to log loss after learning: {e}")
        # ===========================================================

        return {
            'average_loss': average_loss,
            'valid_samples': valid_samples,
            'batch_size': batch_size,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    # -------------------- 其它 --------------------
    def get_performance_summary(self):
        """获取性能摘要"""
        return {
            'decision_history_size': len(self.decision_history),
            'performance_buffer_size': len(self.performance_buffer),
            'recent_success_rate': self._compute_recent_success_rate(),
            'decision_consistency': self._compute_decision_consistency(list(self.decision_history)[-50:]),
            'performance_trend': self._compute_performance_trend(),
            **self.performance_stats
        }

    def save_model(self, filepath):
        """保存GRPO模型（会把历史特征转换为可序列化的列表）"""
        # 把 decision_history 中的 tensor 转为 list，以减少设备依赖
        serial_history = []
        for entry in list(self.decision_history):
            n1, n2, choice, outcome = entry

            def to_serial(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().tolist()
                return x

            serial_history.append((to_serial(n1), to_serial(n2), choice, outcome))

        serial_perf = list(self.performance_buffer)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'decision_history': serial_history,
            'performance_buffer': serial_perf,
            'performance_stats': self.performance_stats,
            'recent_decisions': self.recent_decisions
        }, filepath)
        print(f"GRPO model saved to {filepath}")

    def load_model(self, filepath):
        """加载GRPO模型（会尝试把序列化的历史还原为 tensor）"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            # 加载参数和优化器（注意：优化器 state 可能包含 CUDA tensors，如果在 CPU 上加载会自动 map）
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 恢复历史数据（将 list 转回 tensor）
            serial_history = checkpoint.get('decision_history', [])
            restored = deque(maxlen=self.history_size)
            for entry in serial_history:
                n1, n2, choice, outcome = entry
                rn1 = torch.tensor(n1, dtype=torch.float32) if isinstance(n1, (list, tuple)) else n1
                rn2 = torch.tensor(n2, dtype=torch.float32) if isinstance(n2, (list, tuple)) else n2
                restored.append((rn1, rn2, choice, outcome))

            self.decision_history = restored
            self.performance_buffer = deque(checkpoint.get('performance_buffer', []), maxlen=self.history_size)
            self.performance_stats = checkpoint.get('performance_stats', self.performance_stats)
            self.recent_decisions = checkpoint.get('recent_decisions', [])

            print(f"GRPO model loaded from {filepath}")
            print(f"Restored {len(self.decision_history)} decision records")

        except Exception as e:
            print(f"Error loading GRPO model: {e}")

    def reset_learning(self):
        """重置学习状态 - 用于开始新的学习阶段"""
        self.decision_history.clear()
        self.performance_buffer.clear()
        self.recent_decisions.clear()
        self.performance_stats = {
            'total_decisions': 0,
            'correct_predictions': 0,
            'average_confidence': 0.0,
            'decision_diversity': 0.0
        }
        print("GRPO learning state reset")

    def get_decision_analysis(self, node1, node2, current_board_state, tcn_prediction=None, time_step=0):
        """
        详细分析两个节点的决策过程 - 用于调试和理解
        不实际做决策，只返回分析结果
        """

        self.eval()
        with torch.no_grad():
            # 计算所有特征
            node1_features = self.extract_node_features(node1, time_step)
            node2_features = self.extract_node_features(node2, time_step)

            encoded_node1 = self.node_encoder(node1_features)
            encoded_node2 = self.node_encoder(node2_features)

            comparison_features = self.compute_comparison_features(node1_features, node2_features)
            context_features = self.compute_group_context()
            temporal_features = self.compute_temporal_features(current_board_state, tcn_prediction, time_step)

            # 分别计算各部分的贡献
            combined_nodes = torch.cat([encoded_node1, encoded_node2])
            comparison_input = torch.cat([combined_nodes, comparison_features])
            comparison_score = self.comparison_network(comparison_input)

            context_encoding = self.context_encoder(context_features)
            temporal_encoding = self.temporal_analyzer(temporal_features)

            fusion_input = torch.cat([comparison_score.flatten(), context_encoding, temporal_encoding])
            decision_prob = self.decision_fusion(fusion_input)

            # 详细分析结果
            analysis = {
                'node1_features': {
                    'obj_score': getattr(node1, 'obj', 0.0),
                    'depth': getattr(node1, 'depth', 0),
                    'visits': len(getattr(node1, 'children', [])),
                    'feature_vector': node1_features.tolist()
                },
                'node2_features': {
                    'obj_score': getattr(node2, 'obj', 0.0),
                    'depth': getattr(node2, 'depth', 0),
                    'visits': len(getattr(node2, 'children', [])),
                    'feature_vector': node2_features.tolist()
                },
                'comparison_analysis': {
                    'feature_similarity': F.cosine_similarity(node1_features.unsqueeze(0),
                                                              node2_features.unsqueeze(0), dim=1)[0].item(),
                    'obj_advantage': 'node1' if getattr(node1, 'obj', 0) > getattr(node2, 'obj', 0) else 'node2',
                    'depth_advantage': 'node1' if getattr(node1, 'depth', 0) > getattr(node2, 'depth', 0) else 'node2',
                    'comparison_score': comparison_score.item()
                },
                'context_influence': {
                    'decision_history_size': len(self.decision_history),
                    'recent_success_rate': self._compute_recent_success_rate(),
                    'decision_consistency': self._compute_decision_consistency(list(self.decision_history)[-10:]),
                    'context_score': float(torch.mean(context_features).item())
                },
                'temporal_influence': {
                    'temporal_consistency': float(temporal_features[-3].item() if len(temporal_features) > 2 else 0.0),
                    'tcn_prediction_available': tcn_prediction is not None,
                    'time_pressure': min(1.0, time_step / 100.0)
                },
                'final_decision': {
                    'decision_probability': decision_prob.item(),
                    'recommended_choice': 'node1' if decision_prob.item() > 0.5 else 'node2',
                    'confidence': abs(decision_prob.item() - 0.5) * 2,
                    'certainty_level': 'high' if abs(decision_prob.item() - 0.5) > 0.3 else 'medium' if abs(
                        decision_prob.item() - 0.5) > 0.15 else 'low'
                }
            }

            return analysis


class GRPOGameManager:
    """
    GRPO游戏管理器 - 管理整个游戏过程中的GRPO学习
    """

    def __init__(self, position_dim=8, history_size=1000, learning_rate=0.001):
        self.grpo_decider = GRPOBinaryDecisionMaker(position_dim, history_size, learning_rate)
        self.game_outcomes = []  # 记录每局游戏的结果
        self.current_game_decisions = []  # 当前游戏的决策序列
        self.learning_interval = 50  # 每50个决策学习一次

    def make_game_decision(self, node1, node2, chessboard, tcn_output=None, turn_number=0):
        """在游戏中做决策"""
        selected_node, confidence, decision_info = self.grpo_decider.make_decision(
            node1=node1,
            node2=node2,
            current_board_state=chessboard,
            tcn_prediction=tcn_output,
            time_step=turn_number
        )

        # 记录当前游戏的决策
        self.current_game_decisions.append({
            'turn': turn_number,
            'selected_node': selected_node,
            'confidence': confidence,
            'decision_info': decision_info
        })

        return selected_node, confidence, decision_info

    def end_game(self, game_result):
        """
        游戏结束时的处理

        Args:
            game_result: 游戏结果 (1: 胜利, 0: 失败, 0.5: 平局)
        """

        # 记录游戏结果
        self.game_outcomes.append(game_result)

        # 为本局游戏的所有决策提供反馈
        for i, decision in enumerate(self.current_game_decisions):
            # 基于游戏结果和决策时间给出反馈分数
            # 早期决策的影响更大
            time_weight = 1.0 - (i / len(self.current_game_decisions)) * 0.3
            outcome_score = game_result * time_weight + (1 - time_weight) * 0.5

            self.grpo_decider.update_decision_outcome(outcome_score)

        # 清空当前游戏决策记录
        self.current_game_decisions.clear()

        # 检查是否需要学习
        if len(self.grpo_decider.decision_history) % self.learning_interval == 0:
            learning_result = self.grpo_decider.learn_from_batch(
                batch_size=min(32, len(self.grpo_decider.decision_history) // 4))
            if learning_result:
                print(f"GRPO Post-Game Learning - Loss: {learning_result['average_loss']:.4f}")

        # 打印游戏统计
        if len(self.game_outcomes) % 10 == 0:
            recent_winrate = sum(self.game_outcomes[-10:]) / 10
            print(f"GRPO Recent 10 games win rate: {recent_winrate:.2f}")

            performance = self.grpo_decider.get_performance_summary()
            print(f"GRPO Performance - Success Rate: {performance['recent_success_rate']:.3f}, "
                  f"Consistency: {performance['decision_consistency']:.3f}")

    def get_detailed_analysis(self, node1, node2, chessboard, tcn_output=None, turn_number=0):
        """获取详细的决策分析"""
        return self.grpo_decider.get_decision_analysis(
            node1, node2, chessboard, tcn_output, turn_number
        )

    def save_progress(self, filepath):
        """保存学习进度"""
        self.grpo_decider.save_model(filepath)

        # 额外保存游戏管理器的状态
        manager_state = {
            'game_outcomes': self.game_outcomes,
            'learning_interval': self.learning_interval
        }

        torch.save(manager_state, filepath.replace('.pth', '_manager.pth'))

    def load_progress(self, filepath):
        """加载学习进度"""
        self.grpo_decider.load_model(filepath)

        try:
            manager_state = torch.load(filepath.replace('.pth', '_manager.pth'))
            self.game_outcomes = manager_state['game_outcomes']
            self.learning_interval = manager_state['learning_interval']
        except Exception:
            print("Manager state not found, using defaults")


# 使用示例和集成到原有代码的接口
def integrate_grpo_decision(original_newnode, original_anothernode, chessboard, tcn_output=None, turns=0):
    """
    集成GRPO决策到原有代码中的接口函数
    替代原来的简单random.choices选择
    """

    # 全局GRPO游戏管理器（在实际使用中应该是类的成员变量）
    if not hasattr(integrate_grpo_decision, 'grpo_manager'):
        integrate_grpo_decision.grpo_manager = GRPOGameManager(
            position_dim=8,
            history_size=1000,
            learning_rate=0.001
        )

    grpo_manager = integrate_grpo_decision.grpo_manager

    # 使用GRPO做决策
    selected_node, confidence, decision_info = grpo_manager.make_game_decision(
        node1=original_anothernode,
        node2=original_newnode,
        chessboard=chessboard,
        tcn_output=tcn_output,
        turn_number=turns
    )

    # 打印决策信息（可选）
    print(f"GRPO Turn {turns}: {decision_info['choice_reason']}")
    print(
        f"Confidence: {confidence:.3f}, Node1 obj: {decision_info['node1_obj']:.3f}, Node2 obj: {decision_info['node2_obj']:.3f}")

    return selected_node, decision_info


def grpo_end_game(game_result):
    """
    游戏结束时调用此函数

    Args:
        game_result: 1 (胜利), 0 (失败), 0.5 (平局)
    """
    if hasattr(integrate_grpo_decision, 'grpo_manager'):
        integrate_grpo_decision.grpo_manager.end_game(game_result)


def get_grpo_analysis(node1, node2, chessboard, tcn_output=None, turns=0):
    """
    获取GRPO的详细决策分析 - 用于调试
    """
    if hasattr(integrate_grpo_decision, 'grpo_manager'):
        return integrate_grpo_decision.grpo_manager.get_detailed_analysis(
            node1, node2, chessboard, tcn_output, turns
        )
    return None


def save_grpo_model(filepath="grpo_model.pth"):
    """保存GRPO模型"""
    if hasattr(integrate_grpo_decision, 'grpo_manager'):
        integrate_grpo_decision.grpo_manager.save_progress(filepath)


def load_grpo_model(filepath="grpo_model.pth"):
    """加载GRPO模型"""
    if not hasattr(integrate_grpo_decision, 'grpo_manager'):
        integrate_grpo_decision.grpo_manager = GRPOGameManager()
    integrate_grpo_decision.grpo_manager.load_progress(filepath)
