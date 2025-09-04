import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
import math
from typing import List, Tuple, Dict, Optional


class GRPOGeneticSelector(nn.Module):
    """
    基于GRPO的遗传算法节点选择器（已修正版）

    关键修复：
    - 修正 diversity_analyzer 输入维度不匹配
    - 统一 device 管理（所有新建 tensor 使用 model device）
    - selection_history 存可回放的特征 tensors（CPU），便于稳定学习
    - learn_from_selections 使用历史特征直接回放训练，确保训练/推理输入一致
    - 修复 gini_coefficient 除零问题
    - 使用 BCELoss 训练概率（与网络 Sigmoid 输出配合）
    """

    def __init__(self,
                 position_dim=8,
                 max_population_size=100,
                 history_size=500,
                 learning_rate=0.001):
        super(GRPOGeneticSelector, self).__init__()

        self.position_dim = position_dim
        self.max_population_size = max_population_size
        self.history_size = history_size

        # 现在 selection_history 保存每次选择时用于回放训练的特征（CPU tensors）
        # 格式: (node_features_cpu, context_encoding_cpu, temporal_features_cpu, population_stats_cpu, outcome)
        self.selection_history = deque(maxlen=history_size)
        self.population_performance = deque(maxlen=history_size)
        self.diversity_history = deque(maxlen=history_size)

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(position_dim + 7, 64),  # position + [obj, depth, visits, win_rate, diversity_score, novelty, time_step]
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 群体多样性分析器 - 输入为 3 x diversity_metrics（每个8） => 24
        self.diversity_analyzer = nn.Sequential(
            nn.Linear(8 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # 输出 8 维 diversity metrics
        )

        # 节点互补性评估器
        self.complementarity_network = nn.Sequential(
            nn.Linear(32 * 2 + 8, 48),  # node1 + node2 + diversity_context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)  # complementarity score
        )

        # 群体上下文编码器
        self.population_context_encoder = nn.Sequential(
            nn.Linear(12, 32),  # population-level features
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # 时序一致性评估器
        self.temporal_consistency_network = nn.Sequential(
            nn.Linear(position_dim * 2 + 4, 24),  # current + tcn_prediction + temporal_info
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, 8),
            nn.ReLU()
        )

        # 最终选择概率计算器
        self.selection_probability_net = nn.Sequential(
            nn.Linear(32 + 16 + 8 + 3, 32),  # node_features + context + temporal + population_stats
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出选择概率
        )

        # 优化器和损失函数（使用 BCELoss 训练概率）
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        # 性能追踪
        self.selection_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'diversity_score': 0.0,
            'novelty_bonus': 0.0,
            'temporal_consistency': 0.0
        }

        self.node_type_patterns = defaultdict(list)

    def _model_device(self):
        return next(self.parameters()).device

    # ---------------- 基础特征提取 ----------------
    def extract_node_features(self, node, population_context, time_step, tcn_prediction=None):
        """提取单个节点的完整特征（返回 tensor 在 model device 上）"""
        try:
            device = self._model_device()

            obj_score = float(getattr(node, 'obj', 0.0))
            depth = float(getattr(node, 'depth', 0))
            visits = float(len(getattr(node, 'children', [])))
            win_rate = float(getattr(node, 'win_rate', 0.5))

            # 位置特征 - 优先使用 node.projection 或 node.board
            if hasattr(node, 'projection') and node.projection is not None:
                position_features = torch.tensor(node.projection, dtype=torch.float32, device=device)
            elif hasattr(node, 'board') and node.board is not None:
                position_features = self._extract_position_from_board(node.board).to(device)
            else:
                position_features = torch.zeros(self.position_dim, dtype=torch.float32, device=device)

            diversity_score = float(self._compute_node_diversity(node, population_context))
            novelty_score = float(self._compute_node_novelty(node))

            tail = torch.tensor([obj_score, depth, visits, win_rate, diversity_score, novelty_score, float(time_step)], dtype=torch.float32, device=device)
            node_features = torch.cat([position_features.float(), tail])

            return node_features
        except Exception as e:
            device = self._model_device()
            print(f"Error extracting node features: {e}")
            return torch.zeros(self.position_dim + 7, device=device)

    def _extract_position_from_board(self, board):
        """从棋盘状态提取位置分布特征，返回 CPU tensor（若用于 model，需 .to(device)）"""
        try:
            if hasattr(board, 'board') and callable(board.board):
                board_array = np.array(board.board())
            elif hasattr(board, '__call__'):
                board_array = np.array(board())
            else:
                board_array = np.array(board) if board is not None else np.zeros((10, 10))

            position_stats = []
            for piece_type in range(1, 5):
                positions = np.where(board_array == piece_type)
                if len(positions[0]) > 0:
                    center_x = np.mean(positions[0]) / 9.0
                    center_y = np.mean(positions[1]) / 9.0
                else:
                    center_x, center_y = 0.5, 0.5
                position_stats.extend([center_x, center_y])

            return torch.tensor(position_stats[:self.position_dim], dtype=torch.float32)
        except Exception as e:
            print(f"Error extracting position: {e}")
            return torch.zeros(self.position_dim)

    # ---------------- 群体统计 ----------------
    def _compute_node_diversity(self, target_node, population):
        try:
            if not population or len(population) <= 1:
                return 1.0

            target_obj = getattr(target_node, 'obj', 0.0)
            target_depth = getattr(target_node, 'depth', 0)

            obj_differences = []
            depth_differences = []
            for other_node in population:
                if other_node is target_node:
                    continue
                other_obj = getattr(other_node, 'obj', 0.0)
                other_depth = getattr(other_node, 'depth', 0)
                obj_differences.append(abs(target_obj - other_obj))
                depth_differences.append(abs(target_depth - other_depth))

            if not obj_differences:
                return 1.0

            avg_obj_diff = np.mean(obj_differences)
            avg_depth_diff = np.mean(depth_differences)
            diversity_score = min(1.0, (avg_obj_diff + avg_depth_diff * 0.1) / 2.0)
            return float(diversity_score)
        except Exception as e:
            print(f"Error computing diversity: {e}")
            return 0.5

    def _compute_node_novelty(self, node):
        try:
            if len(self.selection_history) < 10:
                return 0.5

            node_obj = getattr(node, 'obj', 0.0)
            node_depth = getattr(node, 'depth', 0)

            similarities = []
            # selection_history stores tuples where index 0 is node_features_cpu (we may not have original node)
            for hist in list(self.selection_history)[-50:]:
                try:
                    hist_node_feats = hist[0]
                    # hist_node_feats cpu tensor: last 7 positions include obj, depth at indices -7 and -6? but we stored obj as part of features
                    if isinstance(hist_node_feats, torch.Tensor) and hist_node_feats.numel() >= (self.position_dim + 2):
                        hist_node_feats = hist_node_feats.numpy()
                        hist_obj = float(hist_node_feats[-7]) if len(hist_node_feats) >= (self.position_dim + 7) else 0.0
                        hist_depth = float(hist_node_feats[-6]) if len(hist_node_feats) >= (self.position_dim + 7) else 0.0
                    else:
                        # fallback try to use selection_history stored selected node (not used normally)
                        continue

                    obj_sim = 1.0 - min(1.0, abs(node_obj - hist_obj) / max(0.1, abs(hist_obj)))
                    depth_sim = 1.0 - min(1.0, abs(node_depth - hist_depth) / max(1, hist_depth))
                    similarities.append((obj_sim + depth_sim) / 2.0)
                except Exception:
                    continue

            if not similarities:
                return 0.5

            novelty = 1.0 - np.mean(similarities)
            return max(0.0, min(1.0, novelty))
        except Exception as e:
            print(f"Error computing novelty: {e}")
            return 0.5

    def compute_population_diversity_metrics(self, population):
        """返回 tensor（在 model device 上）"""
        device = self._model_device()
        if not population or len(population) <= 1:
            return torch.zeros(8, device=device)

        try:
            obj_scores = [getattr(node, 'obj', 0.0) for node in population]
            depths = [getattr(node, 'depth', 0) for node in population]
            visits = [len(getattr(node, 'children', [])) for node in population]

            obj_std = float(np.std(obj_scores)) if len(obj_scores) > 1 else 0.0
            depth_std = float(np.std(depths)) if len(depths) > 1 else 0.0
            visit_std = float(np.std(visits)) if len(visits) > 1 else 0.0

            obj_range = float(max(obj_scores) - min(obj_scores)) if obj_scores else 0.0
            depth_range = float(max(depths) - min(depths)) if depths else 0.0

            def gini_coefficient(values):
                if len(values) <= 1:
                    return 0.0
                sorted_values = sorted(values)
                n = len(values)
                cumulative = np.cumsum(sorted_values)
                total = cumulative[-1]
                if total == 0:
                    return 0.0
                return (n + 1 - 2 * np.sum(cumulative) / total) / n

            obj_uniformity = 1.0 - gini_coefficient(obj_scores)
            quality_dispersion = obj_std / max(0.1, float(np.mean(obj_scores)))

            metrics = torch.tensor([
                obj_std, depth_std, visit_std, obj_range,
                depth_range, obj_uniformity, quality_dispersion,
                float(len(population)) / float(self.max_population_size)
            ], dtype=torch.float32, device=device)

            return metrics
        except Exception as e:
            print(f"Error computing population diversity: {e}")
            return torch.zeros(8, device=device)

    def compute_population_context(self, population):
        device = self._model_device()
        try:
            if not population:
                return torch.zeros(12, device=device)

            obj_scores = [getattr(node, 'obj', 0.0) for node in population]
            depths = [getattr(node, 'depth', 0) for node in population]
            visits = [len(getattr(node, 'children', [])) for node in population]

            obj_mean = float(np.mean(obj_scores)) if obj_scores else 0.0
            obj_std = float(np.std(obj_scores)) if len(obj_scores) > 1 else 0.0
            obj_max = float(max(obj_scores)) if obj_scores else 0.0
            obj_min = float(min(obj_scores)) if obj_scores else 0.0

            depth_mean = float(np.mean(depths)) if depths else 0.0
            visit_mean = float(np.mean(visits)) if visits else 0.0

            recent_success_rate = float(self._compute_recent_success_rate())
            selection_consistency = float(self._compute_selection_consistency())
            diversity_trend = float(self._compute_diversity_trend())
            quality_trend = float(self._compute_quality_trend())
            exploration_factor = float(self._compute_exploration_factor())
            exploitation_factor = 1.0 - exploration_factor

            context_features = torch.tensor([
                obj_mean, obj_std, obj_max, obj_min, depth_mean, visit_mean,
                recent_success_rate, selection_consistency, diversity_trend,
                quality_trend, exploration_factor, exploitation_factor
            ], dtype=torch.float32, device=device)

            return context_features
        except Exception as e:
            print(f"Error computing population context: {e}")
            return torch.zeros(12, device=device)

    def _compute_recent_success_rate(self):
        if len(self.population_performance) < 5:
            return 0.5
        recent_performance = list(self.population_performance)[-10:]
        return float(np.mean(recent_performance))

    def _compute_selection_consistency(self):
        if len(self.selection_history) < 10:
            return 0.5
        recent = list(self.selection_history)[-20:]
        obj_prefs = []
        for entry in recent:
            try:
                feats = entry[0]
                if isinstance(feats, torch.Tensor):
                    a = feats.numpy()
                    if len(a) >= (self.position_dim + 7):
                        obj_prefs.append(a[-7])
            except:
                continue
        if len(obj_prefs) < 5:
            return 0.5
        prefs_std = np.std(obj_prefs)
        consistency = 1.0 / (1.0 + prefs_std)
        return min(1.0, consistency)

    def _compute_diversity_trend(self):
        if len(self.diversity_history) < 5:
            return 0.0
        recent = list(self.diversity_history)[-10:]
        if len(recent) < 5:
            return 0.0
        x = np.arange(len(recent))
        try:
            trend = np.polyfit(x, recent, 1)[0]
            return float(np.clip(trend, -1.0, 1.0))
        except:
            return 0.0

    def _compute_quality_trend(self):
        if len(self.population_performance) < 5:
            return 0.0
        recent = list(self.population_performance)[-10:]
        if len(recent) < 5:
            return 0.0
        x = np.arange(len(recent))
        try:
            trend = np.polyfit(x, recent, 1)[0]
            return float(np.clip(trend, -1.0, 1.0))
        except:
            return 0.0

    def _compute_exploration_factor(self):
        recent_success = self._compute_recent_success_rate()
        consistency = self._compute_selection_consistency()
        exploration = 1.0 - (recent_success + consistency) / 2.0
        return max(0.1, min(0.9, exploration))

    # ---------------- 时序特征 ----------------
    def compute_temporal_features(self, current_board, tcn_prediction, time_step):
        device = self._model_device()
        try:
            current_features = self._extract_position_from_board(current_board).to(device)

            if tcn_prediction is not None:
                if isinstance(tcn_prediction, torch.Tensor):
                    tcn_features = tcn_prediction.flatten()[:self.position_dim].to(device)
                else:
                    tcn_features = torch.tensor(tcn_prediction, dtype=torch.float32, device=device)[:self.position_dim]
                if tcn_features.numel() < self.position_dim:
                    tcn_features = F.pad(tcn_features, (0, self.position_dim - tcn_features.numel())).to(device)
            else:
                tcn_features = torch.zeros(self.position_dim, device=device)

            temporal_consistency = F.cosine_similarity(current_features.unsqueeze(0), tcn_features.unsqueeze(0), dim=1)[0]
            position_deviation = torch.norm(current_features - tcn_features)
            prediction_confidence = 1.0 / (1.0 + position_deviation)
            time_pressure = torch.tensor(min(1.0, time_step / 100.0), dtype=torch.float32, device=device)

            temporal_features = torch.cat([
                current_features,
                tcn_features,
                temporal_consistency.unsqueeze(0),
                position_deviation.unsqueeze(0),
                prediction_confidence.unsqueeze(0),
                time_pressure.unsqueeze(0)
            ])

            return temporal_features
        except Exception as e:
            print(f"Error computing temporal features: {e}")
            return torch.zeros(self.position_dim * 2 + 4, device=device)

    # ---------------- 选择主流程 ----------------
    def select_from_population(self, population, current_board, tcn_prediction=None, time_step=0, top_k=1):
        if not population:
            return [], {'error': 'Empty population'}
        if len(population) == 1:
            return population, {'single_candidate': True}

        self.eval()
        with torch.no_grad():
            device = self._model_device()

            population_context = self.compute_population_context(population)
            context_encoding = self.population_context_encoder(population_context)

            diversity_metrics = self.compute_population_diversity_metrics(population)
            # 这里我们把相同 metrics 复制 3 次作为输入
            diversity_encoding = self.diversity_analyzer(torch.cat([diversity_metrics, diversity_metrics, diversity_metrics]))

            temporal_features = self.compute_temporal_features(current_board, tcn_prediction, time_step)
            temporal_encoding = self.temporal_consistency_network(temporal_features)

            node_probabilities = []
            node_features_list = []
            population_stats_list = []

            for node in population:
                try:
                    node_features = self.extract_node_features(node, population, time_step, tcn_prediction)
                    node_encoded = self.node_encoder(node_features)
                    node_features_list.append(node_features)

                    obj_score = float(getattr(node, 'obj', 0.0))
                    depth = float(getattr(node, 'depth', 0))
                    visits = float(len(getattr(node, 'children', [])))
                    population_stats = torch.tensor([obj_score, depth, visits], dtype=torch.float32, device=device)
                    population_stats_list.append(population_stats)

                    fusion_input = torch.cat([
                        node_encoded,           # 32
                        context_encoding,       # 16
                        temporal_encoding,      # 8
                        population_stats        # 3
                    ])

                    selection_prob = self.selection_probability_net(fusion_input)
                    node_probabilities.append(float(selection_prob.item()))
                except Exception as e:
                    print(f"Error processing node: {e}")
                    node_probabilities.append(0.0)
                    node_features_list.append(torch.zeros(self.position_dim + 7, device=device))
                    population_stats_list.append(torch.zeros(3, device=device))

            adjusted_probabilities = self._apply_diversity_bonus(node_probabilities, population, node_features_list)

            # 选择
            if top_k == 1:
                try:
                    probs = np.array(adjusted_probabilities)
                    if probs.sum() <= 0:
                        selected_idx = int(np.argmax(probs)) if len(probs) > 0 else 0
                    else:
                        selected_idx = int(np.random.choice(len(population), p=probs / probs.sum()))
                    selected_nodes = [population[selected_idx]]
                except Exception:
                    selected_idx = int(np.argmax(adjusted_probabilities))
                    selected_nodes = [population[selected_idx]]
            else:
                top_indices = np.argsort(adjusted_probabilities)[-top_k:][::-1]
                selected_nodes = [population[i] for i in top_indices]

            # 记录选择历史（保存可回放的特征 tensors 到 CPU）
            # 我们针对每个被选中的节点保存一条记录
            for sel_node in selected_nodes:
                try:
                    # 找到对应 index
                    idx = population.index(sel_node)
                    nf = node_features_list[idx].detach().cpu()
                    ce = context_encoding.detach().cpu()
                    tf = temporal_features.detach().cpu()
                    ps = population_stats_list[idx].detach().cpu()
                    # 存为 (node_features_cpu, context_cpu, temporal_cpu, population_stats_cpu, outcome)
                    self.selection_history.append((nf, ce, tf, ps, None))
                except Exception:
                    # 备用记录
                    self.selection_history.append((torch.zeros(self.position_dim + 7), torch.zeros(16), torch.zeros(self.position_dim * 2 + 4), torch.zeros(3), None))

            # 更新统计
            self._update_selection_stats(adjusted_probabilities, float(diversity_metrics.mean().item()))
            self.diversity_history.append(float(diversity_metrics.mean().item()))

            selection_info = {
                'population_size': len(population),
                'selected_count': len(selected_nodes),
                'selection_probabilities': adjusted_probabilities,
                'diversity_score': float(diversity_metrics.mean().item()),
                'temporal_consistency': float(temporal_features[-4].item()) if temporal_features.numel() > 3 else 0.0,
                'exploration_factor': float(self._compute_exploration_factor()),
                'top_node_obj': float(getattr(selected_nodes[0], 'obj', 0.0)) if selected_nodes else 0.0,
                'population_quality_mean': float(np.mean([getattr(n, 'obj', 0.0) for n in population])),
                'selection_reasoning': self._generate_selection_reasoning(selected_nodes[0] if selected_nodes else None, population, adjusted_probabilities[np.argmax(adjusted_probabilities)] if adjusted_probabilities else 0.0)
            }

            return selected_nodes, selection_info

    def _apply_diversity_bonus(self, base_probabilities, population, node_features_list):
        try:
            exploration_factor = self._compute_exploration_factor()
            diversity_bonuses = []
            for i, node in enumerate(population):
                diversity_score = self._compute_node_diversity(node, population)
                novelty_score = self._compute_node_novelty(node)
                diversity_bonus = exploration_factor * (diversity_score + novelty_score) / 2.0
                diversity_bonuses.append(diversity_bonus)

            adjusted = [base_probabilities[i] * (1.0 + diversity_bonuses[i]) for i in range(len(base_probabilities))]
            total = sum(adjusted)
            if total > 0:
                adjusted = [p / total for p in adjusted]
            else:
                adjusted = [1.0 / len(base_probabilities)] * len(base_probabilities)
            return adjusted
        except Exception as e:
            print(f"Error applying diversity bonus: {e}")
            return base_probabilities

    def _update_selection_stats(self, probabilities, diversity_score):
        self.selection_stats['total_selections'] += 1
        self.selection_stats['diversity_score'] = (self.selection_stats['diversity_score'] * 0.9 + diversity_score * 0.1)
        if probabilities:
            entropy = -sum(p * math.log(p + 1e-8) for p in probabilities if p > 0)
            max_entropy = math.log(len(probabilities)) if len(probabilities) > 0 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            self.selection_stats['novelty_bonus'] = (self.selection_stats['novelty_bonus'] * 0.9 + normalized_entropy * 0.1)

    def _generate_selection_reasoning(self, selected_node, population, selection_prob):
        if not selected_node or not population:
            return "无有效选择"
        reasons = []
        obj_score = getattr(selected_node, 'obj', 0.0)
        population_objs = [getattr(n, 'obj', 0.0) for n in population]
        if obj_score >= np.percentile(population_objs, 75):
            reasons.append("高质量节点")
        elif obj_score >= np.percentile(population_objs, 50):
            reasons.append("中等质量节点")
        else:
            reasons.append("探索性选择")
        diversity_score = self._compute_node_diversity(selected_node, population)
        if diversity_score > 0.7:
            reasons.append("高多样性")
        elif diversity_score > 0.4:
            reasons.append("适度多样性")
        else:
            reasons.append("收敛选择")
        if selection_prob > 0.8:
            reasons.append("高置信度")
        elif selection_prob > 0.6:
            reasons.append("中等置信度")
        else:
            reasons.append("低置信度")
        return "; ".join(reasons)

    # ---------------- 结果反馈与学习 ----------------
    def update_selection_outcome(self, outcome_score):
        if self.selection_history:
            last = self.selection_history[-1]
            # last structure: (node_feat_cpu, context_cpu, temporal_cpu, pop_stats_cpu, outcome)
            self.selection_history[-1] = (last[0], last[1], last[2], last[3], outcome_score)
        self.population_performance.append(outcome_score)
        if outcome_score > 0.6:
            self.selection_stats['successful_selections'] += 1

    def learn_from_selections(self, batch_size=16):
        if len(self.selection_history) < batch_size:
            return None
        self.train()
        valid = [s for s in self.selection_history if s[4] is not None]
        if len(valid) < batch_size:
            return None
        batch = random.sample(valid, batch_size)
        total_loss = 0.0
        valid_samples = 0
        for node_feat_cpu, context_cpu, temporal_cpu, pop_stats_cpu, outcome in batch:
            try:
                device = self._model_device()
                node_features = node_feat_cpu.to(device)
                context_encoding = context_cpu.to(device)
                temporal_features = temporal_cpu.to(device)
                population_stats = pop_stats_cpu.to(device)

                node_encoded = self.node_encoder(node_features)
                # 保证维度一致
                fusion_input = torch.cat([node_encoded, context_encoding[:16], self.temporal_consistency_network(temporal_features)[:8], population_stats])
                predicted_prob = self.selection_probability_net(fusion_input)

                # 基于 outcome 调整目标概率
                if outcome > 0.7:
                    target_prob = min(1.0, float(predicted_prob.item()) + 0.2)
                elif outcome < 0.3:
                    target_prob = max(0.0, float(predicted_prob.item()) - 0.2)
                else:
                    target_prob = float(predicted_prob.item())

                target = torch.tensor([target_prob], dtype=predicted_prob.dtype, device=predicted_prob.device)
                loss = self.criterion(predicted_prob, target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += float(loss.item())
                valid_samples += 1
            except Exception as e:
                print(f"Error in learning step: {e}")
                continue
        average_loss = total_loss / max(1, valid_samples)
        return {'average_loss': average_loss, 'valid_samples': valid_samples, 'batch_size': batch_size, 'total_selections': len(self.selection_history)}

    def get_selection_summary(self):
        success_rate = (self.selection_stats['successful_selections'] / max(1, self.selection_stats['total_selections']))
        return {
            'total_selections': self.selection_stats['total_selections'],
            'success_rate': success_rate,
            'average_diversity': self.selection_stats['diversity_score'],
            'novelty_bonus': self.selection_stats['novelty_bonus'],
            'recent_performance': self._compute_recent_success_rate(),
            'selection_consistency': self._compute_selection_consistency(),
            'exploration_factor': self._compute_exploration_factor(),
            'history_size': len(self.selection_history)
        }

    def save_selector(self, filepath):
        # serialise selection_history (tensors -> lists)
        serial_history = []
        for entry in list(self.selection_history):
            nf, ce, tf, ps, outcome = entry
            def to_serial(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().tolist()
                return x
            serial_history.append((to_serial(nf), to_serial(ce), to_serial(tf), to_serial(ps), outcome))

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'selection_history': serial_history,
            'population_performance': list(self.population_performance),
            'diversity_history': list(self.diversity_history),
            'selection_stats': self.selection_stats,
            'node_type_patterns': dict(self.node_type_patterns)
        }, filepath)
        print(f"GRPO Genetic Selector saved to {filepath}")

    def load_selector(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            serial = checkpoint.get('selection_history', [])
            restored = deque(maxlen=self.history_size)
            for nf, ce, tf, ps, outcome in serial:
                rnf = torch.tensor(nf, dtype=torch.float32) if isinstance(nf, (list, tuple)) else nf
                rce = torch.tensor(ce, dtype=torch.float32) if isinstance(ce, (list, tuple)) else ce
                rtf = torch.tensor(tf, dtype=torch.float32) if isinstance(tf, (list, tuple)) else tf
                rps = torch.tensor(ps, dtype=torch.float32) if isinstance(ps, (list, tuple)) else ps
                restored.append((rnf, rce, rtf, rps, outcome))

            self.selection_history = restored
            self.population_performance = deque(checkpoint.get('population_performance', []), maxlen=self.history_size)
            self.diversity_history = deque(checkpoint.get('diversity_history', []), maxlen=self.history_size)
            self.selection_stats = checkpoint.get('selection_stats', self.selection_stats)
            self.node_type_patterns = defaultdict(list, checkpoint.get('node_type_patterns', {}))

            print(f"GRPO Genetic Selector loaded from {filepath}")
            print(f"Restored {len(self.selection_history)} selection records")
        except Exception as e:
            print(f"Error loading selector: {e}")


class GRPOGeneticManager:
    def __init__(self, position_dim=8, max_population_size=100, learning_rate=0.001):
        self.genetic_selector = GRPOGeneticSelector(position_dim=position_dim, max_population_size=max_population_size, learning_rate=learning_rate)
        self.game_selections = []
        self.learning_interval = 30

    def select_nodes(self, population, chessboard, tcn_prediction=None, time_step=0, top_k=1):
        selected_nodes, selection_info = self.genetic_selector.select_from_population(population=population, current_board=chessboard, tcn_prediction=tcn_prediction, time_step=time_step, top_k=top_k)
        self.game_selections.append({'time_step': time_step, 'population_size': len(population), 'selected_nodes': selected_nodes, 'selection_info': selection_info})
        return selected_nodes, selection_info

    def end_game(self, game_result):
        for i, selection in enumerate(self.game_selections):
            time_weight = 1.0 - (i / len(self.game_selections)) * 0.2
            outcome_score = game_result * time_weight + (1 - time_weight) * 0.5
            self.genetic_selector.update_selection_outcome(outcome_score)
        self.game_selections.clear()
        if (self.genetic_selector.selection_stats['total_selections'] % self.learning_interval == 0):
            learning_result = self.genetic_selector.learn_from_selections(batch_size=16)
            if learning_result:
                print(f"GRPO Genetic Learning - Loss: {learning_result['average_loss']:.4f}")
        if self.genetic_selector.selection_stats['total_selections'] % 100 == 0:
            summary = self.genetic_selector.get_selection_summary()
            print(f"GRPO Genetic Stats - Success Rate: {summary['success_rate']:.3f}, Diversity: {summary['average_diversity']:.3f}")

    def get_analysis(self, population, chessboard, tcn_prediction=None, time_step=0):
        if not population:
            return {'error': 'Empty population'}
        obj_scores = [getattr(node, 'obj', 0.0) for node in population]
        depths = [getattr(node, 'depth', 0) for node in population]
        visits = [len(getattr(node, 'children', [])) for node in population]
        diversity_metrics = self.genetic_selector.compute_population_diversity_metrics(population)
        context = self.genetic_selector.compute_population_context(population)
        analysis = {
            'population_size': len(population),
            'quality_stats': {
                'obj_mean': float(np.mean(obj_scores)),
                'obj_std': float(np.std(obj_scores)),
                'obj_range': [float(min(obj_scores)), float(max(obj_scores))],
                'depth_mean': float(np.mean(depths)),
                'visit_mean': float(np.mean(visits))
            },
            'diversity_metrics': {
                'overall_diversity': float(diversity_metrics.mean().item()),
                'obj_diversity': float(diversity_metrics[0].item()),
                'depth_diversity': float(diversity_metrics[1].item()),
                'quality_dispersion': float(diversity_metrics[6].item())
            },
            'context_analysis': {
                'recent_success_rate': float(context[6].item()),
                'selection_consistency': float(context[7].item()),
                'exploration_factor': float(context[10].item()),
                'exploitation_factor': float(context[11].item())
            },
            'recommendations': self._generate_recommendations(population, diversity_metrics, context)
        }
        return analysis

    def _generate_recommendations(self, population, diversity_metrics, context):
        recommendations = []
        diversity_score = float(diversity_metrics.mean().item())
        exploration_factor = float(context[10].item())
        recent_success = float(context[6].item())
        if diversity_score < 0.3:
            recommendations.append("群体多样性偏低，建议增加探索")
        elif diversity_score > 0.8:
            recommendations.append("群体多样性很高，可以适度收敛")
        if exploration_factor > 0.7:
            recommendations.append("当前处于高探索模式")
        elif exploration_factor < 0.3:
            recommendations.append("当前处于高利用模式")
        if recent_success < 0.4:
            recommendations.append("最近表现不佳，建议调整策略")
        elif recent_success > 0.7:
            recommendations.append("最近表现良好，可以延续当前策略")
        obj_scores = [getattr(node, 'obj', 0.0) for node in population]
        if max(obj_scores) - min(obj_scores) < 0.1:
            recommendations.append("候选质量差异很小，建议基于多样性选择")
        return recommendations

    def save_manager(self, filepath):
        self.genetic_selector.save_selector(filepath)

    def load_manager(self, filepath):
        self.genetic_selector.load_selector(filepath)


# 集成接口
def integrate_grpo_genetic_selection(population, chessboard, tcn_prediction=None, time_step=0):
    if not hasattr(integrate_grpo_genetic_selection, 'grpo_genetic_manager'):
        integrate_grpo_genetic_selection.grpo_genetic_manager = GRPOGeneticManager(position_dim=8, max_population_size=100, learning_rate=0.001)
    manager = integrate_grpo_genetic_selection.grpo_genetic_manager
    selected_nodes, selection_info = manager.select_nodes(population=population, chessboard=chessboard, tcn_prediction=tcn_prediction, time_step=time_step, top_k=1)
    if selected_nodes:
        selected_node = selected_nodes[0]
        print(f"GRPO Genetic Selection at step {time_step}:")
        print(f"  Population size: {selection_info['population_size']}")
        print(f"  Selected obj: {selection_info['top_node_obj']:.4f}")
        print(f"  Population mean obj: {selection_info['population_quality_mean']:.4f}")
        print(f"  Diversity score: {selection_info['diversity_score']:.3f}")
        print(f"  Reasoning: {selection_info['selection_reasoning']}")
        return selected_node, selection_info
    else:
        if population:
            weights = [getattr(node, 'obj', 0.0) for node in population]
            if sum(weights) > 0:
                return random.choices(population, weights=weights)[0], {'fallback': True}
            else:
                return random.choice(population), {'fallback': True}
        else:
            return None, {'error': 'Empty population'}


def grpo_genetic_end_game(game_result):
    if hasattr(integrate_grpo_genetic_selection, 'grpo_genetic_manager'):
        integrate_grpo_genetic_selection.grpo_genetic_manager.end_game(game_result)


def get_grpo_genetic_analysis(population, chessboard, tcn_prediction=None, time_step=0):
    if hasattr(integrate_grpo_genetic_selection, 'grpo_genetic_manager'):
        return integrate_grpo_genetic_selection.grpo_genetic_manager.get_analysis(population, chessboard, tcn_prediction, time_step)
    return None


def save_grpo_genetic_model(filepath="grpo_genetic_model.pth"):
    if hasattr(integrate_grpo_genetic_selection, 'grpo_genetic_manager'):
        integrate_grpo_genetic_selection.grpo_genetic_manager.save_manager(filepath)


def load_grpo_genetic_model(filepath="grpo_genetic_model.pth"):
    if not hasattr(integrate_grpo_genetic_selection, 'grpo_genetic_manager'):
        integrate_grpo_genetic_selection.grpo_genetic_manager = GRPOGeneticManager()
    integrate_grpo_genetic_selection.grpo_genetic_manager.load_manager(filepath)
