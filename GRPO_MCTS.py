import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os


class GRPONodeEvaluator(nn.Module):
    """GRPO网络，用于评估MCTS节点在群体中的相对价值"""

    def __init__(self, node_feature_dim=8, group_feature_dim=11, hidden_dim=128):
        super(GRPONodeEvaluator, self).__init__()

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 群体特征编码器
        self.group_encoder = nn.Sequential(
            nn.Linear(group_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出相对价值调整
        )

        # 探索常数调整网络
        self.exploration_network = nn.Sequential(
            nn.Linear(group_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出[0,1]，用于调整探索常数
        )
        # 立即初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化所有网络权重"""

        def init_module(module):
            if isinstance(module, nn.Linear):
                # 使用He初始化（针对ReLU激活函数优化）
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.Sequential):
                # 递归处理Sequential容器
                for submodule in module:
                    init_module(submodule)

        # 对所有子模块进行初始化
        init_module(self.node_encoder)
        init_module(self.group_encoder)
        init_module(self.fusion_network)
        init_module(self.exploration_network)

        print("[GRPONodeEvaluator] 权重初始化完成")

    def check_parameters(self):
        """检查参数是否正常"""
        total_params = 0
        zero_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            print(f"{name}: shape={param.shape}, zeros={zero_params}/{param.numel()}, "
                  f"min={param.min().item():.6f}, max={param.max().item():.6f}, "
                  f"mean={param.mean().item():.6f}")

        print(f"总参数: {total_params}, 零参数: {zero_params}, 零参数比例: {zero_params / total_params:.2%}")
        return zero_params == 0

    def forward(self, node_features, group_features):
        """
        Args:
            node_features: [batch_size, node_feature_dim] 节点特征
            group_features: [batch_size, group_feature_dim] 群体特征

        Returns:
            relative_value: [batch_size, 1] 相对价值调整
            exploration_factor: [batch_size, 1] 探索因子调整
        """

        node_encoded = self.node_encoder(node_features)
        group_encoded = self.group_encoder(group_features)

        # 特征融合
        fused = torch.cat([node_encoded, group_encoded], dim=-1)
        relative_value = self.fusion_network(fused)

        # 探索调整
        exploration_factor = self.exploration_network(group_features)
        return relative_value, exploration_factor

import os

class GRPOAgent:
    """GRPO代理，用于MCTS节点选择优化"""

    def __init__(self, node_feature_dim=8, group_feature_dim=11, lr=0.001, state_file='grpo_network_state.pth', base_dir='./train'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = GRPONodeEvaluator(node_feature_dim, group_feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # 状态文件路径
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.state_file_path = os.path.join(self.base_dir, state_file)

        # 自动加载已保存的网络状态
        self._auto_load_or_initialize()

        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = 32

        # 训练参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.update_frequency = 50  # 每50次选择更新一次
        self.selection_count = 0

        # 优先尝试加载“全量状态”（network + optimizer + buffer + selection_count）
        loaded_full = False
        try:
            loaded_full = self.load_full_state()
        except Exception as e:
            print(f"尝试加载全量状态时出错: {e}")

    def _auto_load_or_initialize(self):
        """自动加载已保存的网络状态，如果不存在则初始化并保存"""
        print("=== GRPO网络自动加载/初始化 ===")

        if os.path.exists(self.state_file_path):
            try:
                # 尝试加载已保存的状态
                state_dict = torch.load(self.state_file_path, map_location=self.device)
                self.network.load_state_dict(state_dict)
                print(f"✓ 成功加载网络状态: {self.state_file_path}")

                # 验证加载的参数是否正常
                if self._check_parameters_valid():
                    print("✓ 加载的网络参数验证通过")
                    return
                else:
                    print("⚠ 加载的网络参数异常（可能全零），将重新初始化")

            except Exception as e:
                print(f"⚠ 加载网络状态失败: {e}，将重新初始化")
        else:
            print(f"✓ 网络状态文件不存在: {self.state_file_path}")
            print("✓ 将创建新的网络状态")

        # 如果加载失败或文件不存在，则初始化并保存
        print("正在初始化网络参数...")
        self.network.initialize_weights()  # 使用网络自带的初始化方法

        # 验证初始化
        if self._check_parameters_valid():
            print("✓ 网络参数初始化成功")
            # 保存初始化后的状态
            self.save_network_state()
        else:
            print("❌ 网络参数初始化失败，尝试强制重新初始化")
            self.force_reinitialize()

    def _check_parameters_valid(self):
        """检查网络参数是否有效（非全零且数值合理）"""
        try:
            total_params = 0
            zero_params = 0
            invalid_params = 0

            for param in self.network.parameters():
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
                # 检查是否有异常值（NaN或无穷大）
                invalid_params += (~torch.isfinite(param)).sum().item()

            zero_ratio = zero_params / total_params if total_params > 0 else 1.0
            invalid_ratio = invalid_params / total_params if total_params > 0 else 0.0

            print(f"参数统计: 总数={total_params}, 零参数={zero_params}({zero_ratio:.1%}), "
                  f"异常参数={invalid_params}({invalid_ratio:.1%})")

            # 如果零参数比例过高（>90%）或存在无效参数，则认为无效
            return zero_ratio < 0.9 and invalid_ratio == 0.0

        except Exception as e:
            print(f"参数检查错误: {e}")
            return False

    def save_network_state(self):
        """保存当前网络状态"""
        try:
            torch.save(self.network.state_dict(), self.state_file_path)
            print(f"✓ 网络状态已保存: {self.state_file_path}")
            return True
        except Exception as e:
            print(f"❌ 保存网络状态失败: {e}")
            return False

    def force_reinitialize(self):
        """强制重新初始化所有参数"""
        print("执行强制重新初始化...")

        def apply_strong_init(module):
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化确保非零
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

        self.network.apply(apply_strong_init)

        # 重新创建优化器
        lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0.001
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        print("强制重新初始化完成")
        if self._check_parameters_valid():
            print("✓ 强制初始化成功")
            self.save_network_state()
        else:
            print("❌ 强制初始化仍然失败")

    def update_and_save_network(self):
        """更新网络并自动保存状态"""
        self._update_network()
        # 定期保存网络状态（每100次更新保存一次）
        if self.selection_count % (self.update_frequency * 2) == 0:
            self.save_network_state()

    def extract_node_features(self, feasible):
        """
        从feasible中提取节点特征
        feasible格式: [原位置，现位置, mode, val, 上一个节点]
        """
        try:
            # feasible结构解析
            origin_pos = feasible[0]  # 原位置
            current_pos = feasible[1]  # 现位置
            mode = feasible[2]  # 模式
            val = feasible[3]  # 价值
            prev_node = feasible[4]  # 上一个节点

            # 基础特征
            obj_value = float(val) if hasattr(val, '__float__') else 0.0
            mode_value = int(mode) if hasattr(mode, '__float__') else 0.0
            visit_count = float(prev_node.time) if hasattr(prev_node, 'time') else 1.0

            # 位置特征
            pos_features = self._extract_position_features(origin_pos, current_pos)

            # 节点结构特征
            node_structure_features = self._extract_node_structure_features(prev_node)

            features = [
                obj_value,  # 节点价值
                mode_value,  # 模式（黑棋/白棋）
                visit_count,  # 访问次数
                *pos_features,  # 位置相关特征 (4维)
                *node_structure_features  # 节点结构特征 (1维)
            ]

            return torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"特征提取错误: {e}")
            return torch.zeros(8, dtype=torch.float32)

    def _extract_position_features(self, origin_pos, current_pos):
        """
        提取位置相关特征

        Args:
            origin_pos: 原位置
            current_pos: 现位置

        Returns:
            position_features: [移动距离, 位置重要性, 位置变化方向, 位置类型]
        """
        try:
            # 计算移动距离
            if hasattr(origin_pos, '__len__') and hasattr(current_pos, '__len__'):
                origin = np.array(origin_pos) if not isinstance(origin_pos, np.ndarray) else origin_pos
                current = np.array(current_pos) if not isinstance(current_pos, np.ndarray) else current_pos

                move_distance = float(np.linalg.norm(current - origin)) if origin.shape == current.shape else 0.0

                # 位置重要性（距离中心的距离）
                center = np.array([5, 5])  # 假设10x10棋盘的中心
                pos_importance = float(np.linalg.norm(current - center)) if len(current) >= 2 else 0.0

                # 移动方向（简化为水平/垂直/对角线）
                if len(origin) >= 2 and len(current) >= 2:
                    dx, dy = current[0] - origin[0], current[1] - origin[1]
                    if abs(dx) > abs(dy):
                        move_direction = 1.0  # 水平移动
                    elif abs(dy) > abs(dx):
                        move_direction = 2.0  # 垂直移动
                    else:
                        move_direction = 3.0  # 对角线移动
                else:
                    move_direction = 0.0

                # 位置类型（边缘/中心/角落）
                if len(current) >= 2:
                    x, y = current[0], current[1]
                    if (x == 0 or x == 9) and (y == 0 or y == 9):
                        pos_type = 1.0  # 角落
                    elif x == 0 or x == 9 or y == 0 or y == 9:
                        pos_type = 2.0  # 边缘
                    else:
                        pos_type = 3.0  # 中心
                else:
                    pos_type = 0.0

                return [move_distance, pos_importance, move_direction, pos_type]
            else:
                return [0.0, 0.0, 0.0, 0.0]

        except Exception as e:
            print(f"位置特征提取错误: {e}")
            return [0.0, 0.0, 0.0, 0.0]

    def _extract_node_structure_features(self, prev_node):
        """
        提取节点结构特征

        Args:
            prev_node: 上一个节点对象

        Returns:
            structure_features: [节点深度估计]
        """
        try:
            # 基于访问次数估计节点在树中的相对深度
            visit_count = float(prev_node.time) if hasattr(prev_node, 'time') else 1.0
            estimated_depth = np.log2(visit_count + 1)  # 基于访问次数的深度估计

            return [estimated_depth]
        except Exception as e:
            print(f"节点结构特征提取错误: {e}")
            return [0.0]

    def analyze_group_state(self, all_feasible_nodes):
        """
        分析当前候选节点群体的状态
        all_feasible_nodes格式: [[原位置，现位置, mode, val, 上一个节点], ...]
        """
        if not all_feasible_nodes:
            return torch.zeros(11, dtype=torch.float32)

        # 提取所有节点的关键信息
        val_values = []  # 节点价值
        visit_counts = []  # 访问次数
        modes = []  # 模式分布
        move_distances = []  # 移动距离

        for feasible in all_feasible_nodes:
            try:
                # feasible结构: [原位置，现位置, mode, val, 上一个节点]
                origin_pos = feasible[0]
                current_pos = feasible[1]
                mode = feasible[2]
                val = feasible[3]
                prev_node = feasible[4]

                # 提取数值
                val_value = float(val) if hasattr(val, '__float__') else 0.0
                visit_count = float(prev_node.time) if hasattr(prev_node, 'time') else 1.0
                mode_value = float(mode) if hasattr(mode, '__float__') else 0.0

                # 计算移动距离
                if hasattr(origin_pos, '__len__') and hasattr(current_pos, '__len__'):
                    origin = np.array(origin_pos)
                    current = np.array(current_pos)
                    move_dist = float(np.linalg.norm(current - origin)) if origin.shape == current.shape else 0.0
                else:
                    move_dist = 0.0

                val_values.append(val_value)
                visit_counts.append(visit_count)
                modes.append(mode_value)
                move_distances.append(move_dist)

            except Exception as inner_e:
                print(f"单个节点解析错误: {inner_e}")
                val_values.append(0.0)
                visit_counts.append(1.0)
                modes.append(0.0)
                move_distances.append(0.0)

        # 转换为numpy数组进行统计计算
        val_values = np.array(val_values)
        visit_counts = np.array(visit_counts)
        modes = np.array(modes)
        move_distances = np.array(move_distances)
        # 计算群体统计特征
        group_features = [
            float(np.mean(val_values)),  # val均值
            float(np.std(val_values) + 1e-6),  # val标准差
            float(np.max(val_values) - np.min(val_values)),  # val范围
            float(np.mean(visit_counts)),  # 访问均值
            float(np.std(visit_counts) + 1e-6),  # 访问标准差
            float(len(all_feasible_nodes)),  # 群体大小
            float(self._calculate_entropy(visit_counts)),  # 访问熵（多样性指标）
            float(self._calculate_gini(val_values)),  # 价值基尼系数（一致性指标）
            float(np.mean(move_distances)),  # 平均移动距离
            float(np.std(modes)),  # 模式分布标准差
            float(np.sum(visit_counts)),  # 总访问次数
        ]

        return torch.tensor(group_features, dtype=torch.float32)

    def _calculate_entropy(self, values):
        """计算数组的熵"""
        try:
            values = np.array(values) + 1e-6  # 避免log(0)
            probs = values / np.sum(values)
            return -np.sum(probs * np.log(probs + 1e-6))
        except:
            return 0.0

    def _calculate_gini(self, values):
        """计算基尼系数"""
        try:
            values = np.sort(np.array(values))
            n = len(values)
            cumsum = np.cumsum(values)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        except:
            return 0.0

    def evaluate_node_in_group(self, node_features, group_context, search_context):
        """
        评估节点在群体中的相对价值

        Args:
            node_features: 单个节点的特征
            group_context: 当前搜索群体的状态
            search_context: 搜索上下文信息

        Returns:
            relative_value: 相对价值调整量
        """
        try:
            self.network.eval()

            with torch.no_grad():
                # 确保输入是正确的张量格式
                if isinstance(node_features, dict):
                    node_tensor = self._dict_to_tensor(node_features)
                else:
                    node_tensor = torch.tensor(node_features, dtype=torch.float32)

                if isinstance(group_context, dict):
                    group_tensor = self._dict_to_tensor(group_context)
                else:
                    group_tensor = torch.tensor(group_context, dtype=torch.float32)

                # 确保是二维张量 [1, features]
                if node_tensor.dim() == 1:
                    node_tensor = node_tensor.unsqueeze(0)
                if group_tensor.dim() == 1:
                    group_tensor = group_tensor.unsqueeze(0)

                # 移到设备
                node_tensor = node_tensor.to(self.device)
                group_tensor = group_tensor.to(self.device)

                # 网络评估
                relative_value, exploration_factor = self.network(node_tensor, group_tensor)

                return relative_value.cpu().item()

        except Exception as e:
            print(f"GRPO评估错误: {e}")
            return 0.0

    def _dict_to_tensor(self, feature_dict):
        """将特征字典转换为张量"""
        if isinstance(feature_dict, dict):
            return torch.tensor(list(feature_dict.values()), dtype=torch.float32)
        return torch.tensor(feature_dict, dtype=torch.float32)

    def select_best_node(self, all_feasible_nodes, base_ucb_values):
        """
        从候选节点中选择最优节点

        Args:
            all_feasible_nodes: 所有候选节点列表
            base_ucb_values: 对应的基础UCB值列表

        Returns:
            selected_index: 选中的节点索引
            adjusted_values: GRPO调整后的所有节点价值
        """
        if not all_feasible_nodes:
            return 0, []

        try:
            # 分析群体状态
            group_state = self.analyze_group_state(all_feasible_nodes)

            # 为每个节点计算GRPO调整
            adjusted_values = []

            for i, node in enumerate(all_feasible_nodes):
                # 提取节点特征
                node_features = self.extract_node_features(node)

                # GRPO评估
                search_context = {'node_index': i, 'total_candidates': len(all_feasible_nodes)}
                relative_value = self.evaluate_node_in_group(
                    node_features, group_state, search_context
                )

                # 调整UCB值
                adjusted_ucb = base_ucb_values[i] + relative_value
                adjusted_values.append(adjusted_ucb)

            # 选择策略：epsilon-greedy with softmax
            if random.random() < self.epsilon:
                # 探索：随机选择
                selected_index = random.randint(0, len(all_feasible_nodes) - 1)
            else:
                # 利用：选择最高价值的节点
                selected_index = np.argmax(adjusted_values)

            # 记录选择经验
            self._record_selection_experience(
                all_feasible_nodes, group_state, selected_index, adjusted_values
            )

            self.selection_count += 1

            # 定期更新网络
            if self.selection_count % self.update_frequency == 0:
                self._update_network()

            return selected_index, adjusted_values

        except Exception as e:
            print(f"节点选择错误: {e}")
            # 回退到随机选择
            return random.randint(0, len(all_feasible_nodes) - 1), base_ucb_values

    def _record_selection_experience(self, nodes, group_state, selected_idx, values):
        """记录选择经验到经验池"""
        try:
            experience = {
                'group_state': group_state.clone() if torch.is_tensor(group_state) else group_state,
                'node_features': self.extract_node_features(nodes[selected_idx]),
                'selected_index': selected_idx,
                'predicted_values': values.copy() if isinstance(values, list) else values,
                'timestamp': self.selection_count
            }
            self.experience_buffer.append(experience)
        except Exception as e:
            print(f"经验记录错误: {e}")

    def update_experience_rewards(self, game_result, final_best_node_quality):
        """根据对弈结果更新经验池中的奖励"""
        try:
            # 计算奖励衰减
            current_time = self.selection_count

            for exp in reversed(list(self.experience_buffer)):
                if current_time - exp['timestamp'] > 1000:  # 只更新最近的经验
                    break

                # 时间衰减
                time_decay = self.gamma ** (current_time - exp['timestamp'])

                # 计算奖励
                immediate_reward = self._calculate_immediate_reward(exp)
                long_term_reward = game_result * time_decay

                exp['reward'] = immediate_reward + long_term_reward
                exp['final_quality'] = final_best_node_quality

        except Exception as e:
            print(f"奖励更新错误: {e}")

    def _calculate_immediate_reward(self, experience):
        """计算即时奖励"""
        try:
            # 基于选择的节点在群体中的相对表现
            predicted_values = experience['predicted_values']
            selected_idx = experience['selected_index']

            if isinstance(predicted_values, list) and len(predicted_values) > selected_idx:
                selected_value = predicted_values[selected_idx]
                avg_value = np.mean(predicted_values)
                max_value = np.max(predicted_values)

                # 奖励公式：选中节点的相对优势
                relative_advantage = (selected_value - avg_value) / (max_value - avg_value + 1e-6)
                return float(relative_advantage)

            return 0.0
        except:
            return 0.0

    def _update_network(self):
        """使用经验池更新GRPO网络"""
        if len(self.experience_buffer) < self.batch_size:
            return

        try:
            self.network.train()

            # 采样批次
            batch = random.sample(list(self.experience_buffer), self.batch_size)

            # 准备训练数据
            states = []
            group_states = []
            rewards = []

            for exp in batch:
                if 'reward' in exp:  # 只使用已计算奖励的经验
                    states.append(exp['node_features'])
                    group_states.append(exp['group_state'])
                    rewards.append(exp['reward'])

            if len(states) < 5:  # 需要足够的训练样本
                return

            # 转换为张量
            state_tensor = torch.stack(states).to(self.device)
            group_tensor = torch.stack(group_states).to(self.device)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

            # 前向传播
            predicted_values, exploration_factors = self.network(state_tensor, group_tensor)
            predicted_values = predicted_values.squeeze()

            # 计算损失（均方误差）
            loss = F.mse_loss(predicted_values, reward_tensor)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            print(f"GRPO网络更新 - Loss: {loss.item():.6f}")

            # 返回 loss 值，供外部记录使用
            return float(loss.item())

        except Exception as e:
            print(f"网络更新错误: {e}")
            return None

    def adjust_exploration_constant(self, base_c, group_diversity, consensus_level, search_progress):
        """动态调整探索常数"""
        try:
            self.network.eval()

            # 构建群体特征用于探索调整
            group_features = torch.tensor([
                group_diversity,
                consensus_level,
                search_progress,
                base_c,
                float(len(self.experience_buffer)),
                float(self.selection_count),
                0.0, 0.0, 0.0, 0.0, 0.0  # 补齐到11维
            ], dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, exploration_factor = self.network(
                    torch.zeros(1, 8).to(self.device),  # 占位符
                    group_features
                )

                # 调整C值：factor在[0,1]，我们映射到[0.5, 2.0]
                c_multiplier = 0.5 + 1.5 * exploration_factor.item()
                adjusted_c = base_c * c_multiplier

                return max(0.1, min(5.0, adjusted_c))  # 限制范围

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"探索常数调整错误: {e}")
            return base_c

    # --------- 以下为新增/替换的方法，用于保存/加载全量状态 ----------
    def _serialize_experience(self, exp):
        import pickle
        """把一个经验条目转成可 pickle 的类型（把 tensor -> list）"""
        e = {}
        for k, v in exp.items():
            if isinstance(v, torch.Tensor):
                e[k] = v.cpu().tolist()
            elif isinstance(v, (list, tuple)):
                e_list = []
                for vv in v:
                    if isinstance(vv, torch.Tensor):
                        e_list.append(vv.cpu().tolist())
                    else:
                        e_list.append(vv)
                e[k] = e_list
            else:
                # 如果是不可序列化的自定义对象（比如 prev_node），尽量保存其可重建字段
                # 这里保守处理：直接尝试 pickle，如失败则跳过该字段
                try:
                    pickle.dumps(v)
                    e[k] = v
                except Exception:
                    # 可选：只保留部分字段（例如时间戳/位置等），具体依你的对象而定
                    e[k] = v  # 如果这导致问题，你可以在这里改为更安全的映射
        return e

    def _deserialize_experience(self, exp):
        """把保存的经验反序列化回可用结构（list -> torch.Tensor）"""
        e = {}
        for k, v in exp.items():
            # 如果是数值列表，转为 tensor；其它保持不变
            if isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v):
                try:
                    e[k] = torch.tensor(v, dtype=torch.float32)
                except Exception:
                    e[k] = v
            else:
                e[k] = v
        return e

    def save_full_state(self, filename='grpo_MCTS_full_state.pth'):
        """保存 network + optimizer + experience_buffer + selection_count 到单个文件"""
        try:
            path = os.path.join(self.base_dir, filename)
            payload = {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'selection_count': self.selection_count,
                # 把 deque 变成 list 并序列化每个元素
                'experience_buffer': [self._serialize_experience(exp) for exp in list(self.experience_buffer)]
            }
            # 使用 torch.save（内部用 pickle），也可以换成 pickle.dump
            torch.save(payload, path)
            print(f"✓ 全量状态已保存: {path}")
            return True
        except Exception as e:
            print(f"❌ 保存全量状态失败: {e}")
            return False

    def load_full_state(self, filename='grpo_MCTS_full_state.pth'):
        """从文件加载完整状态（如存在）"""
        try:
            path = os.path.join(self.base_dir, filename)
            # 如果文件不存在，视情况创建初始文件
            if not os.path.exists(path):
                print(f"⚠ 全量状态文件不存在: {path}")

                print("将创建初始全量状态文件（使用当前网络/优化器/经验缓冲区）...")
                # 确保目录存在
                os.makedirs(self.base_dir, exist_ok=True)

                payload = {
                    'network': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict() if hasattr(self,
                                                                        'optimizer') and self.optimizer is not None else {},
                    'selection_count': self.selection_count,
                    # 把 deque 变成 list 并序列化每个元素
                    'experience_buffer': [self._serialize_experience(exp) for exp in list(self.experience_buffer)]
                }

                try:
                    torch.save(payload, path)
                    print(f"✓ 初始全量状态已创建并保存: {path}")
                    return True
                except Exception as e:
                    print(f"❌ 创建初始全量状态失败: {e}")
                    return False

            payload = torch.load(path, map_location=self.device)

            # 加载 network / optimizer
            if 'network' in payload:
                self.network.load_state_dict(payload['network'])
            if 'optimizer' in payload and hasattr(self, 'optimizer'):
                # 注意：如果模型结构发生变化，可能会出错
                self.optimizer.load_state_dict(payload['optimizer'])

            # restore selection_count
            self.selection_count = payload.get('selection_count', self.selection_count)

            # restore experience_buffer
            saved_buf = payload.get('experience_buffer', [])
            deserialized = [self._deserialize_experience(exp) for exp in saved_buf]
            self.experience_buffer = deque(deserialized, maxlen=getattr(self.experience_buffer, 'maxlen', 10000))

            print(f"✓ 已从 {path} 加载全量状态")
            return True

        except Exception as e:
            print(f"❌ 加载全量状态失败: {e}")
            return False


import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class SimpleGRPOTrainer:
    """简化的GRPO训练器 - 适用于在线单步训练"""

    def __init__(self, grpo_agent):
        self.grpo_agent = grpo_agent
        self.device = grpo_agent.device

        # 训练参数
        self.train_frequency = 10  # 每10次选择训练一次
        self.min_buffer_size = 20  # 最小经验池大小

        # 临时存储当前选择的信息
        self.current_selection = None

        # 日志文件名（保存在 grpo_agent.base_dir 下）
        self.log_filename = "grpo_training_log.txt"

    def _append_log(self, loss=None, reward=None, predicted_value=None, extra=None, filename=None):
        """把一条训练记录追加到日志文件（JSON Lines，每行一个 JSON 对象）"""
        try:
            fname = filename or self.log_filename
            path = os.path.join(self.grpo_agent.base_dir, fname)
            entry = {
                "utc": datetime.utcnow().isoformat() + "Z",
                "selection_count": getattr(self.grpo_agent, "selection_count", None),
                "buffer_size": len(self.grpo_agent.experience_buffer),
                "loss": None if loss is None else float(loss),
                "reward": None if reward is None else float(reward),
                "predicted_value": None if predicted_value is None else float(predicted_value),
                "epsilon": float(getattr(self.grpo_agent, "epsilon", 0.0))
            }
            if isinstance(extra, dict):
                entry.update(extra)

            # 以追加模式写入，一行一个 JSON
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"写日志失败: {e}")

    def record_selection(self, feasible_nodes, selected_node, selected_idx, base_ucb_values=None):
        """
        记录一次节点选择

        Args:
            feasible_nodes: 所有候选节点列表
            selected_node: 被选择的节点
            selected_idx: 选择的索引
            base_ucb_values: 基础UCB值（可选）
        """
        try:
            # 提取特征
            node_features = self.grpo_agent.extract_node_features(selected_node)
            group_features = self.grpo_agent.analyze_group_state(feasible_nodes)

            # 使用GRPO网络预测价值（用于对比学习）
            with torch.no_grad():
                predicted_value, _ = self.grpo_agent.network(
                    node_features.unsqueeze(0).to(self.device),
                    group_features.unsqueeze(0).to(self.device)
                )

            # 存储选择信息
            self.current_selection = {
                'node_features': node_features,
                'group_features': group_features,
                'selected_idx': selected_idx,
                'predicted_value': predicted_value.cpu().item(),
                'feasible_count': len(feasible_nodes),
                'base_ucb_values': base_ucb_values or [],
                'timestamp': self.grpo_agent.selection_count

            }

            self.grpo_agent.selection_count += 1
            print(f"已记录第 {self.grpo_agent.selection_count} 次选择")

        except Exception as e:
            print(f"记录选择错误: {e}")
            self.current_selection = None

    def train_with_reward(self, reward):
        """
        使用奖励训练网络

        Args:
            reward: 奖励值（通常是 new_node.obj 或其他评估值）
        """
        if self.current_selection is None:
            print("警告: 没有记录的选择信息，无法训练")
            return

        try:
            # 标准化奖励
            normalized_reward = self._normalize_reward(reward)

            # 创建训练样本
            experience = {
                'node_features': self.current_selection['node_features'],
                'group_features': self.current_selection['group_features'],
                'reward': normalized_reward,
                'predicted_value': self.current_selection['predicted_value'],
                'timestamp': self.current_selection['timestamp']
            }

            # 添加到经验池
            self.grpo_agent.experience_buffer.append(experience)

            print(f"添加训练样本: 奖励={reward:.4f} (标准化={normalized_reward:.4f}), "
                  f"预测值={self.current_selection['predicted_value']:.4f}")

            loss = self._train_network()
            print(f"网络训练完成，损失: {loss:.6f}")

            # 记录训练日志（在清除 current_selection 之前记录 predicted_value）
            self._append_log(loss=loss,
                             reward=normalized_reward,
                             predicted_value=self.current_selection.get('predicted_value', None),
                             extra={
                                 "feasible_count": self.current_selection.get('feasible_count', None),
                                 "base_ucb_mean": float(
                                     np.mean(self.current_selection['base_ucb_values'])) if self.current_selection.get(
                                     'base_ucb_values') else None
                             })

            # 清除当前选择记录
            self.current_selection = None
            self.grpo_agent.save_full_state()

        except Exception as e:
            print(f"训练错误: {e}")
            import traceback
            traceback.print_exc()

    def _normalize_reward(self, reward):
        """标准化奖励值"""
        try:
            # 简单的标准化：限制在[-1, 1]范围内
            return float(np.clip(reward / 100.0, -1.0, 1.0))
        except:
            return 0.0

    def _train_network(self):
        """训练网络"""
        try:
            self.grpo_agent.network.train()

            # 采样批次
            buffer_list = list(self.grpo_agent.experience_buffer)
            batch_size = min(self.grpo_agent.batch_size, len(buffer_list))
            batch = random.sample(buffer_list, batch_size)

            # 准备数据
            node_features = torch.stack([exp['node_features'] for exp in batch]).to(self.device)
            group_features = torch.stack([exp['group_features'] for exp in batch]).to(self.device)
            target_rewards = torch.tensor([exp['reward'] for exp in batch],
                                          dtype=torch.float32).to(self.device)

            # 前向传播
            predicted_values, exploration_factors = self.grpo_agent.network(node_features, group_features)
            predicted_values = predicted_values.squeeze()

            # 计算损失
            value_loss = F.mse_loss(predicted_values, target_rewards)

            # 反向传播
            self.grpo_agent.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.grpo_agent.network.parameters(), max_norm=1.0)
            self.grpo_agent.optimizer.step()

            self.grpo_agent.network.eval()
            return value_loss.item()

        except Exception as e:
            print(f"网络训练错误: {e}")
            return 0.0

    def get_training_stats(self):
        """获取训练统计信息"""
        buffer_size = len(self.grpo_agent.experience_buffer)
        recent_rewards = []

        if buffer_size > 0:
            recent_experiences = list(self.grpo_agent.experience_buffer)[-10:]
            recent_rewards = [exp['reward'] for exp in recent_experiences]

        stats = {
            'selection_count': self.selection_count,
            'buffer_size': buffer_size,
            'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'recent_reward_std': np.std(recent_rewards) if recent_rewards else 0.0,
            'epsilon': self.grpo_agent.epsilon
        }

        return stats


# 使用示例和集成方法
def integrate_grpo_training_example():
    """展示如何将GRPO训练集成到现有代码中"""

    # 假设你已经有了GRPO代理
    # self.GRPO_MCTS = GRPOAgent(node_feature_dim=8, group_feature_dim=11, lr=1e-3)

    # 创建训练器
    # trainer = SimpleGRPOTrainer(self.GRPO_MCTS)

    # ========== 在你的MCTS选择过程中使用 ==========

    # 1. 在选择节点之前记录信息
    def your_node_selection_function(feasible_nodes, trainer):
        """你的节点选择函数示例"""

        # 使用GRPO选择节点
        base_ucb_values = [calculate_ucb_value(node) for node in feasible_nodes]
        selected_idx, adjusted_values = trainer.grpo_agent.select_best_node(
            feasible_nodes, base_ucb_values
        )

        selected_node = feasible_nodes[selected_idx]

        # 记录选择信息
        trainer.record_selection(
            feasible_nodes=feasible_nodes,
            selected_node=selected_node,
            selected_idx=selected_idx,
            base_ucb_values=base_ucb_values
        )

        return selected_node

    # 2. 在获得奖励后进行训练
    def after_getting_reward(new_node, reward, trainer):
        """获得奖励后的训练"""

        # 使用奖励训练网络
        trainer.train_with_reward(reward)

        # 可选：定期打印统计信息
        if trainer.selection_count % 50 == 0:
            stats = trainer.get_training_stats()
            print(f"\n=== 训练统计 (第 {stats['selection_count']} 次选择) ===")
            print(f"经验池大小: {stats['buffer_size']}")
            print(f"最近平均奖励: {stats['recent_avg_reward']:.4f}")
            print(f"奖励标准差: {stats['recent_reward_std']:.4f}")
            print(f"探索率: {stats['epsilon']:.4f}")

    return """
    集成到你的代码中的步骤：

    1. 在类初始化中添加：
       self.grpo_trainer = SimpleGRPOTrainer(self.GRPO_MCTS)

    2. 在节点选择函数中：
       # 选择节点
       selected_node = your_selection_logic(feasible_nodes)

       # 记录选择
       self.grpo_trainer.record_selection(feasible_nodes, selected_node, selected_idx)

    3. 在获得奖励后：
       # reward 可以是 new_node.obj 或其他评估值
       self.grpo_trainer.train_with_reward(reward)

    4. 定期保存模型：
       if step_count % 100 == 0:
           self.GRPO_MCTS.save_network_state()
    """


def calculate_ucb_value(node):
    """示例UCB值计算函数"""
    # 这里应该是你的UCB计算逻辑
    return random.uniform(0.1, 2.0)


# 完整的使用示例
class YourMCTSWithGRPO:
    """展示如何在MCTS类中集成GRPO训练的完整示例"""

    def __init__(self):
        # 初始化GRPO
        self.GRPO_MCTS = GRPOAgent(node_feature_dim=8, group_feature_dim=11, lr=1e-3)

        # 初始化训练器
        self.grpo_trainer = SimpleGRPOTrainer(self.GRPO_MCTS)

        print("MCTS with GRPO initialized")

    def select_node(self, feasible_nodes):
        """选择节点的主函数"""
        if not feasible_nodes:
            return None

        try:
            # 计算基础UCB值
            base_ucb_values = [self.calculate_ucb(node) for node in feasible_nodes]

            # 使用GRPO选择节点
            selected_idx, adjusted_values = self.GRPO_MCTS.select_best_node(
                feasible_nodes, base_ucb_values
            )

            selected_node = feasible_nodes[selected_idx]

            # 记录选择信息用于训练
            self.grpo_trainer.record_selection(
                feasible_nodes, selected_node, selected_idx, base_ucb_values
            )

            print(
                f"选择节点 {selected_idx}, UCB调整: {base_ucb_values[selected_idx]:.3f} -> {adjusted_values[selected_idx]:.3f}")

            return selected_node

        except Exception as e:
            print(f"节点选择错误: {e}")
            return feasible_nodes[0]  # 回退到第一个节点

    def update_with_reward(self, reward):
        """使用奖励更新GRPO网络"""
        try:
            self.grpo_trainer.train_with_reward(reward)

            # 定期打印统计
            if self.grpo_trainer.selection_count % 25 == 0:
                stats = self.grpo_trainer.get_training_stats()
                print(f"训练进度: {stats['selection_count']} 次选择, "
                      f"经验池: {stats['buffer_size']}, "
                      f"平均奖励: {stats['recent_avg_reward']:.3f}")

            # 定期保存模型
            if self.grpo_trainer.selection_count % 100 == 0:
                self.GRPO_MCTS.save_network_state()
                print("GRPO模型已保存")

        except Exception as e:
            print(f"奖励更新错误: {e}")

    def calculate_ucb(self, node):
        """计算UCB值（你需要根据实际情况实现）"""
        # 示例实现
        return random.uniform(0.5, 2.0)




def get_feasible_nodes():
    """示例：获取候选节点"""
    # 这应该是你现有的逻辑
    return []


def execute_move_and_get_reward(node):
    """示例：执行移动并获取奖励"""
    # 这应该是你现有的逻辑
    # 返回 new_node.obj 或其他奖励值
    return random.uniform(-1, 1)

