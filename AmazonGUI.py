#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import traceback

from Flexible_TCN import TimeClassified, flexibleTCN
from chessboard import ChessBoard
from AiTree import *
import copy
import asyncio
import time
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox, QMdiArea, QAction, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QPalette

# GCN

import warnings
from chessboard import rate

from GRPO_final import *
from GRPO_gene import *
from GRPO_MCTS import *
from policy import *
from models.TCN import flexibleTCN
from models.time_classified import TimeClassified
import action

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from openai import OpenAI

api_key = 'sk-IxoidByfpUpfmhVbMiDdH98oclTSufnOIT4lLGyjEPjoyLzK'
client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")

sys.setrecursionlimit(3000)  # 修改递归深度为 3000
WIDTH = 540
HEIGHT = 540
MARGIN = 22
GRID = (WIDTH - 2 * MARGIN) / (15 - 1)
PIECE = 34
EMPTY = 0
BLACK = 1
WHITE = 2
Block = 3
threshold = 6
variable12 = 3

LEARNING_RATE = 0.01  # 学习率 学习率过小→ →→收敛过慢，学习率过大→ →→错过局部最优；
WEIGHT_DACAY = 5e-4  # 正则化系数 weight_dacay，解决过拟合问题
EPOCHS = 200  # 完整遍历训练集的次数
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"  # 指定设备，如果当前显卡忙于其他工作，可以设置为 DEVICE = "cpu"，使用cpu运行
random.seed(time.time())


def dataconstruct(fournodes, adjacency, array, turn, arrayencoder1, arrayencoder2, newlist):
    idbase = list(range(2, adjacency[0].size()[0] + 1))
    temp = torch.zeros(1, 7, dtype=torch.float32)
    # 更新训练数据array
    for i in range(4):
        # 一阶训练的数据集 + 更新adjacency]
        fournodes[i].postorder_traversal(array, turn, [], 1, idbase, adjacency, arrayencoder1, arrayencoder2, newlist)
        print("here")
        if fournodes[i].mode == 0:
            chess = "黑"
        else:
            chess = "白"

        try:
            content = "下面是一个亚马逊棋的棋盘:\n" + str(
                fournodes[
                    i].board) + "\n其中，1代表白棋，2代表黑棋，3代表一个阻挡。\n你要保证对于1棋子和2棋子来说，两类棋子的评分总和应该为1。注意，你给分应当比较激进。Instruction: 请给出当前局面" + chess + "棋对应的0到1的评分，用于评估" + chess + "棋在该局面下的好坏,分数越高，局面越好。只输出评分，不要输出多余的信息："
            val = fournodes[i].mode
            while 1:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": ""
                        },
                        {
                            "role": "user",
                            "content": content

                        }
                    ],
                    temperature=1,
                    max_tokens=256,
                    top_p=1,
                )
                try:
                    print("main")
                    val = float(response.choices[0].message.content.strip())
                    break
                except:
                    print("mistake")
                    print(str(response.choices[0].message.content))
                    continue
            temp = torch.add(temp, torch.tensor(
                [fournodes[i].weight[0], fournodes[i].weight[1], fournodes[i].weight[2], fournodes[i].weight[3],
                 fournodes[i].weight[4], 1, val], dtype=torch.float32))
            array[0][0, :] = temp / 4
            if torch.all(array[0][0, :-2] == 0):
                array[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0.5], dtype=torch.float32)
                arrayencoder1[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0.5], dtype=torch.float32)
                arrayencoder2[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0.5], dtype=torch.float32)
        except:
            array[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0.5], dtype=torch.float32)
            arrayencoder1[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0.5], dtype=torch.float32)
            arrayencoder2[0][0, :] = torch.tensor([0, 0, 0, 0, 0, 0.5], dtype=torch.float32)

        # try:
        #     val = fournodes[i].mode
        #     temp = torch.add(temp, torch.tensor(
        #         [fournodes[i].weight[0], fournodes[i].weight[1], fournodes[i].weight[2], fournodes[i].weight[3],
        #          fournodes[i].weight[4], 1, val], dtype=torch.float32))
        #     array[0][0, :] = temp / 4
        #     if torch.all(array[0][0, :-2] == 0):
        #         array[0][0, :] = torch.tensor([0, 0, 0, 0, 0, turn, 0.5], dtype=torch.float32)
        # except:
        #     array[0][0, :] = torch.tensor([0, 0, 0, 0, 0, turn, 0.5], dtype=torch.float32)


def diffrence(list1, list2):
    set1 = set(tuple(sublist) for sublist in list1)
    set2 = set(tuple(sublist) for sublist in list2)
    diff = [list(subtuple) for subtuple in (set1 - set2)] + [list(subtuple) for subtuple in (set2 - set1)]
    return diff


def del_children(fournode):
    for node in fournode:
        del_children_recursive(node)
        node.time = 0


def count_children(node):
    # 计算当前节点的子节点数量
    total_children = len(node.children)

    # 递归计算所有子节点的子节点数量
    for child in node.children:
        total_children += count_children(child)

    return total_children


def total_children_count(node_list):
    total_count = 0
    for node in node_list:
        total_count += count_children(node)
    return total_count


def del_children_recursive(node):
    for child in node.children:
        del_children_recursive(child)
    node.children.clear()
    del node


# 定义线程类执行AI的算法
# ----------------------------------------------------------------------
################################
###  GAT LAYER DEFINITION    ###
################################


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 n_heads: int, concat: bool = False, dropout: float = 0.4,
                 leaky_relu_slope: float = 0.2, cuda=None):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads  # Number of attention heads
        self.concat = concat  # whether to concatenate the final attention heads
        self.dropout = dropout  # Dropout rate
        self.cuda = cuda
        if concat:  # concatenating the attention heads
            self.out_features = out_features  # Number of output features per node
            assert out_features % n_heads == 0  # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else:  # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W

        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads), dtype=torch.float32))

        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1), dtype=torch.float32))
        if self.cuda:
            self.W.to(self.cuda)
            self.a.to(self.cuda)
        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)  # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1)  # softmax activation function to the attention coefficients
        self.reset_parameters()  # Reset the parameters

    def reset_parameters(self):

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def _get_attention_scores(self, h_transformed: torch.Tensor):

        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def change_W(self, W):
        self.W = W
        return

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        n_nodes = h.shape[0]
        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)

        # getting the attention scores
        # output shape (n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask)  # masked attention scores

        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (n_nodes, out_features)
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime


class autoencoder(nn.Module):
    def __init__(self, weight):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(5, 3),
                                     nn.LeakyReLU(negative_slope=5e-2)
                                     )
        self.decoder = nn.Sequential(nn.Linear(3, 5),
                                     nn.Tanh()
                                     )
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        encode = self.encoder(x.to("cpu"))
        decode = self.decoder(encode)
        return torch.matmul(decode, self.weight.t().to("cpu"))

    def forward1(self, x):
        encode = self.encoder(x.to("cpu"))
        decode = self.decoder(encode)
        return torch.matmul(decode, self.weight.t().to("cpu"))

    def forward2(self, x):
        encode = self.encoder(x.to(DEVICE))
        decode = self.decoder(encode)
        return decode


class GAT(nn.Module):
    def __init__(self,
                 in_features,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat=False,
                 dropout=0.4,
                 leaky_relu_slope=0.2, cuda=None):
        super(GAT, self).__init__()
        self.cuda = cuda
        # Define the Graph Attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope, cuda=self.cuda
        )
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope, cuda=self.cuda
        )
        self.tanh = nn.Tanh()
        self.autoencoder1 = autoencoder(weight=None)
        self.autoencoder2 = autoencoder(weight=None)

    def forward(self, input_tensor: torch.Tensor, adj_mat: torch.Tensor):
        x = self.autoencoder1.forward2(input_tensor) * rate + self.autoencoder2.forward2(input_tensor) * (1 - rate)
        # Apply the first Graph Attention layer
        x = self.gat1(x, adj_mat)
        x = F.elu(x)  # Apply ELU activation function to the output of the first layer
        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)
        return (self.tanh(x) + 1) / 2  # Apply softmax activation function

    def load_W_a(self, turn):
        self.load_state_dict(torch.load('GAT.pth', map_location=DEVICE))

    def save_W_a(self, turn):
        torch.save(self.state_dict(), "GAT.pth")
        for param in self.autoencoder1.parameters():
            param.requires_grad = True
        for param in self.autoencoder2.parameters():
            param.requires_grad = True

    def load_auto(self):
        for param in self.autoencoder1.parameters():
            param.requires_grad = False
        for param in self.autoencoder2.parameters():
            param.requires_grad = False


model = GAT(in_features=5, n_hidden=128, n_heads=16, num_classes=1, cuda=DEVICE)

model.to(DEVICE)
# criterion3 = nn.BCELoss(size_average=True, reduce=True)
criterion3 = nn.BCELoss()
criterion3.to(DEVICE)
weight = torch.randn(1, 5)  # 使用适当的权重初始化
model1 = autoencoder(weight)
model1.load_state_dict(torch.load('autoencoder_model1.pth'))
model2 = autoencoder(weight)
model2.load_state_dict(torch.load('autoencoder_model2.pth'))
model1.weight = torch.load('weight1.pth')
model2.weight = torch.load('weight2.pth')
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0005)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0005)
criterion1 = nn.MSELoss()
criterion1.to(DEVICE)
model.autoencoder1 = model1
model.autoencoder2 = model2
model.load_state_dict(torch.load('GAT.pth', map_location=DEVICE))
model.autoencoder1 = model1
model.autoencoder2 = model2
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)


# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def projection1(target_node):
    blacknode = copy.deepcopy(target_node.black)
    whitenode = copy.deepcopy(target_node.white)
    sequences = []
    classification = []
    # 造8个
    for i in range(4):
        x, y = projection(blacknode[i].pos[0], blacknode[i].pos[1])
        x, y = add_noise(x, y)
        r = math.sqrt(x ** 2 + y ** 2)
        sequences.append(r)
        classification.append(0)
        x, y = projection(whitenode[i].pos[0], whitenode[i].pos[1])
        x, y = add_noise(x, y)
        r = math.sqrt(x ** 2 + y ** 2)
        sequences.append(r)
        classification.append(1)
    # 排序
    sequences, classification = new_sort(sequences, classification)
    return sequences


def projection(x, y):
    r = max(abs(x), abs(y))
    x = float(x - r / 2)
    y = float(y - r / 2)
    if x == 0 and y == 0:
        return 0.0, 0.0
    return x * float(max(abs(x), abs(y)) / math.sqrt(x ** 2 + y ** 2)), float(
        y * max(abs(x), abs(y)) / math.sqrt(x ** 2 + y ** 2))


def inverse_projection(x, y):
    x1 = x * math.sqrt(x ** 2 + y ** 2) / max(abs(x), abs(y))
    y1 = y * math.sqrt(x ** 2 + y ** 2) / max(abs(x), abs(y))
    return x1 + 5, y1 + 5


def add_noise(x, y):
    dx = random.uniform(-0.5, 0.5)  # 生成 -0.5 到 0.5 之间的随机数
    dy = random.uniform(-0.5, 0.5)  # 生成 -0.5 到 0.5 之间的随机数
    dx = x + dx
    dy = y + dy
    return dx, dy


def update(root):
    pass


def collect_nodes_by_level(node, level, level_dict, visited):
    if not node:
        return

    if level not in level_dict:
        level_dict[level] = []

    # Only collect nodes that are not in the main path
    if node not in visited:
        level_dict[level].append(node.value)

    for child in node.children:
        collect_nodes_by_level(child, level + 1, level_dict, visited)


def find_path_to_headnode(headnodes, targetnode):
    path_to_target = {}  # 存储从目标节点到头节点的路径
    all_nodes_by_level = {}  # 存储所有头节点和子节点，按层级存储

    # 用于存储目标节点的引用
    target_node_ref = None

    def dfs(node, level):
        nonlocal target_node_ref

        # 将当前节点加入到当前层级的字典中
        if level not in all_nodes_by_level:
            all_nodes_by_level[level] = []
        all_nodes_by_level[level].append(node)
        # 检查当前节点是否是目标节点
        if node.number == targetnode.number:
            target_node_ref = node  # 找到目标节点的引用
            return True  # 找到目标节点

        # 遍历子节点
        for child in node.children:
            child.parent = node  # 记录父节点
            if dfs(child, level + 1):
                return True  # 如果找到目标节点，停止递归

        return False  # 没有找到目标节点

    # 遍历所有头节点以寻找目标节点
    for head in headnodes:
        dfs(head, 1)  # 从层级1开始

    # 如果找到了目标节点，构建路径
    if target_node_ref:
        current_node = target_node_ref
        while current_node:
            level = len(path_to_target) + 1  # 计算层级（从1开始）
            path_to_target[level] = path_to_target.get(level, [])
            path_to_target[level].insert(0, current_node)  # 在路径前面插入
            current_node = current_node.parent  # 逐级向上查找

    # 从层级字典中删除路径上的节点
    for level in path_to_target:
        for value in path_to_target[level]:
            # 查找并移除路径上的节点
            for lvl in all_nodes_by_level:
                if value in all_nodes_by_level[lvl]:
                    all_nodes_by_level[lvl].remove(value)

    # 获取路径的最大层级
    max_path_level = len(path_to_target)

    # 只保留不超过路径层级的层级节点
    filtered_nodes_by_level = {lvl: all_nodes_by_level[lvl] for lvl in all_nodes_by_level if lvl <= max_path_level}

    return path_to_target, filtered_nodes_by_level


def new_sort(sequences, classification):
    sorted_pairs = sorted(zip(sequences, classification))

    # 解压排序后的元组
    sorted_sequences, sorted_classification = zip(*sorted_pairs)

    # 将结果转换为列表
    sorted_sequences = list(sorted_sequences)
    sorted_classification = list(sorted_classification)
    return sorted_sequences, sorted_classification


def split_consecutive(lst):
    if not lst:
        return []

    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            current_group.append(lst[i])
        else:
            result.append(current_group)
            current_group = [lst[i]]

    result.append(current_group)  # 添加最后一组
    return result


def coverage(list1, radius):
    # list1是0,1序列 [0,1,1,0,0,1,0,1] result = [[0],[1,1],[0,0],[1],[0],[1]], length=[1,2,2,1,1,1]
    result = split_consecutive(list1)
    count = 0
    length = [len(i) for i in result]
    temp1 = []
    for i in range(len(length)):
        if i == 0 or i == len(length) - 1:  # 对于第一个和最后一个元素
            temp1.append(1)  # 直接添加 1
        else:
            if length[i] <= length[i - 1] and length[i] <= length[i + 1]:
                temp1.append(1)
            else:
                temp1.append(0)

    ratezero = 0
    rateone = 0
    for num in range(len(length)):
        if num == 0:
            if result[num][0] == 0:
                ratezero += radius[count]
            elif result[num][0] == 1:
                rateone += radius[count]
            count += length[num]
        elif num == len(length) - 1:
            if result[num][0] == 0:
                ratezero += (5 - radius[count])
            elif result[num][0] == 1:
                rateone += (5 - radius[count])
            count += length[num]
        elif temp1[num] == 1:
            if length[num] > 1:
                if result[num][0] == 0:  # todo 报错
                    rateone += radius[count + length[num]] - radius[count - 1]
                    ratezero += radius[count + length[num] - 1] - radius[count - 1]
                elif result[num][0] == 1:
                    ratezero += radius[count + length[num]] - radius[count - 1]
                    rateone += radius[count + length[num] - 1] - radius[count - 1]
                count += length[num]
            elif length[num] == 1:
                if result[num][0] == 0:
                    ratezero += radius[count] - radius[count - 1]
                elif result[num][0] == 1:
                    rateone += radius[count] - radius[count - 1]
                count += length[num]

        elif temp1[num] == 0:
            if result[num][0] == 0:
                ratezero += radius[count + length[num] - 1] - radius[count - 1]
            elif result[num][0] == 1:
                rateone += radius[count + length[num] - 1] - radius[count - 1]
            count += length[num]
    return ratezero, rateone  # 0的控制域和 1的控制域


def node2coverage(node, pos1, pos2):
    temp = copy.deepcopy(node)

    for node1 in temp.black:
        if node1.pos == pos1:
            node1.pos = pos2
    for node1 in temp.white:
        if node1.pos == pos1:
            node1.pos = pos2

    list1 = temp.black + temp.white
    sequence = []
    radius = []
    for n in range(len(list1)):
        if n <= 3:
            sequence.append(0)
            x, y = projection(list1[n].pos[0], list1[n].pos[1])
            x, y = add_noise(x, y)
            r = math.sqrt(x ** 2 + y ** 2)
            radius.append(r)
        else:
            sequence.append(1)
            x, y = projection(list1[n].pos[0], list1[n].pos[1])
            x, y = add_noise(x, y)
            r = math.sqrt(x ** 2 + y ** 2)
            radius.append(r)
    radius, sequence = new_sort(radius, sequence)
    del temp
    return coverage(sequence, radius)


def fly_algorithm(target_node, sub_path):
    blacknode = copy.deepcopy(target_node.black)
    whitenode = copy.deepcopy(target_node.white)
    basement = []
    sequencebase = []
    for node in sub_path:
        sequences = []
        classification = []
        for i in range(4):
            x, y = projection(node.black[i].pos[0], node.black[i].pos[1])
            x, y = add_noise(x, y)
            r = math.sqrt(x ** 2 + y ** 2)
            sequences.append(r)
            classification.append(0)
            x, y = projection(node.white[i].pos[0], node.white[i].pos[1])
            x, y = add_noise(x, y)
            r = math.sqrt(x ** 2 + y ** 2)
            sequences.append(r)
            classification.append(1)
        sequences, classification = new_sort(sequences, classification)
        basement.append(classification)
        sequencebase.append(classification)
    sequences = []
    classification = []
    # 造8个
    for i in range(4):
        x, y = projection(blacknode[i].pos[0], blacknode[i].pos[1])
        x, y = add_noise(x, y)
        r = math.sqrt(x ** 2 + y ** 2)
        sequences.append(r)
        classification.append(0)
        x, y = projection(whitenode[i].pos[0], whitenode[i].pos[1])
        x, y = add_noise(x, y)
        r = math.sqrt(x ** 2 + y ** 2)
        sequences.append(r)
        classification.append(1)
    # 排序
    sequences, classification = new_sort(sequences, classification)
    score = 0
    for sample in basement:
        for i in range(8):
            score += int(sample[i]) ^ int(classification[i])
    return [classification, sequences], score / len(basement)


def new_algorithm(headnodes, targetnode, mode):
    path_to_target, all_nodes_by_level = find_path_to_headnode(headnodes, targetnode)
    count = 0
    value = 0
    for key in path_to_target.keys():
        temp, score = fly_algorithm(path_to_target[key][0], all_nodes_by_level[key])
        if path_to_target[key][0].parent:
            mini, maxi = optimization(path_to_target[key][0])  # 最小控制域 最大控制域
        else:
            break
        rate0, rate1 = coverage(temp[0], temp[1])  # 两个控制域
        print(maxi, mini, rate0, rate1)
        if score > variable12:  # 控制循环次数
            if mode == 0:
                value += rate0 / mini
            else:
                value += rate1 / mini
        else:
            if mode == 0:
                value += rate0 / maxi
            else:
                value += rate1 / maxi
        print(value)
        count += 1
    return value / count
    # 计算控制域 如果兄弟节点序列不一致的话：计算控制域；否则，少搜索，因为相对固定了说明没有逆转的空间了，设定一个30?


def optimization(node):  # node是当前节点，mode
    print("optimization")

    def searchchildren(board, node, mode, temp=None):
        # node是一个列表
        if mode == 0:
            for num in range(len(node.black)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (node.black[num].pos[0] + m < 0 or node.black[num].pos[
                            1] + n < 0 or
                                   node.black[num].pos[0] + m > 9 or node.black[num].pos[
                                       1] + n > 9):
                            if board[node.black[num].pos[0] + m][node.black[num].pos[1] + n] == EMPTY:
                                temp.append([[node.black[num].pos[0], node.black[num].pos[1]],
                                             [node.black[num].pos[0] + m, node.black[num].pos[1] + n], mode,
                                             -1, node.black[num]])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        elif mode == 1:
            for num in range(len(node.white)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (node.white[num].pos[0] + m < 0 or node.white[num].pos[
                            1] + n < 0 or
                                   node.white[num].pos[0] + m > 9 or node.white[num].pos[
                                       1] + n > 9):
                            if board[node.white[num].pos[0] + m][node.white[num].pos[1] + n] == EMPTY:
                                temp.append([[node.white[num].pos[0], node.white[num].pos[1]],
                                             [node.white[num].pos[0] + m, node.white[num].pos[1] + n], mode,
                                             -1, node.white[num]])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        return temp

    feasible = searchchildren(node.parent.board, node, node.mode, [])
    min = 999
    max = -999
    for sample in feasible:
        mode = sample[2]
        value = node2coverage(node.parent, sample[0], sample[1])  # value记录了0的控制域和1的控制域
        if mode == 0:
            value = value[0]
        elif mode == 1:
            value = value[1]
        if value < min:
            min = value
        if value > max:
            max = value
    return min, max


# ----------------------------------------------------------------------
# 重新定义Label类
# ----------------------------------------------------------------------
class LaBel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)

    def enterEvent(self, e):
        e.ignore()


class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        bar = self.menuBar()
        file = bar.addMenu("File")
        file.addAction("New")
        file.addAction("last chance")
        file.addAction("Tiled")
        file.triggered[QAction].connect(self.windowaction)
        self.setWindowTitle("AmazonChess Starter")

    def windowaction(self, q):

        if q.text() == "New":
            self.ex = GoBang(0)
            app.exec_()
        if q.text() == "cascade":
            self.mdi.cascadeSubWindows()

        if q.text() == "Tiled":
            self.mdi.tileSubWindows()


def check_list(lst, length):
    if not lst:  # 判断列表是否为空
        return 0

    if len(lst) >= length // 2:
        return 1
    return 0


def check_list1(lst, length):
    if not lst:  # 判断列表是否为空
        return 0

    # 判断列表长度
    if len(lst) >= length:
        return 1
    return 0


trainingsequence = []
trainingsequenceblack = []
trainingsequencewhite = []


class GoBang(QWidget):
    def __init__(self, mode=0):
        super().__init__()
        self.first = BLACK if mode == 0 else WHITE
        self.AI = WHITE if mode == 0 else BLACK
        self.put1 = 0
        self.turns = 1
        self.initUI()

    def initUI(self):
        self.put = False
        self.click = False
        self.my_turn = True if self.first == BLACK else False  # 玩家先行
        self.boardforcal = None
        palette1 = QPalette()  # 设置棋盘背景
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('img/board.jpg')))
        self.setPalette(palette1)
        self.setCursor(Qt.OpenHandCursor)  # 鼠标变成手指形状
        self.sound_piece = QSound("sound/luozi.wav")  # 加载落子音效
        self.sound_win = QSound("sound/win.wav")  # 加载胜利音效
        self.sound_defeated = QSound("sound/defeated.wav")  # 加载失败音效
        self.resize(WIDTH, HEIGHT)  # 固定大小 540*540
        self.setMinimumSize(QSize(WIDTH, HEIGHT))
        self.setMaximumSize(QSize(WIDTH, HEIGHT))
        self.setWindowTitle("AmazonChess")  # 窗口名称
        self.setWindowIcon(QIcon('img/pic1.png'))  # 窗口图标
        # self.lb1 = QLabel('            ', self)
        # self.lb1.move(20, 10)
        self.cursor = QPixmap("img/cursor.png")
        self.cursorlabel = QLabel(self)
        self.cursorlabel.setPixmap(self.cursor)
        # self.cursorlabel.setGeometry(0,0,self.cursor.width(),self.cursor.height())
        # self.cursorlabel.show()
        self.cursorlabel.hide()
        self.black = QPixmap('img/black.png')
        self.white = QPixmap('img/white.png')
        self.block = QPixmap('img/block.png')
        # self.cursor = QPixmap('img/cursor.png')
        self.lastx = 0  # 记录上一个数据
        self.lasty = 0  # 记录上一个数据
        self.mouse_point = LaBel(self)  # 鼠标改棋子
        self.pieceswhite = [QLabel(self) for i in range(4)]  # 新建棋子标签，准备在棋盘上绘制棋子
        self.piecesblack = [QLabel(self) for i in range(4)]
        self.piecesblock = [[QLabel(self) for n in range(10)] for m in range(10)]

        self.GRPO_MCTS = GRPOAgent(node_feature_dim=8, group_feature_dim=11, lr=1e-3)
        self.grpo_trainer = SimpleGRPOTrainer(self.GRPO_MCTS)

        self.flexible_tcn_model = flexibleTCN(maxlimit=4, model=TimeClassified(m=3, n=5, lowerlimit=2))
        self.flexible_tcn_model.to(DEVICE)
        self.policy_value_net = PolicyValueNet(position_dim=10, action_space=2700, hidden_dim=128,
                                               dropout=0.3)
        self.policy_value_net.to(DEVICE)

        self.chessboard = ChessBoard(paras=model1, paras2=model2)  # 棋盘类·

        self.blocknum = 0
        for piece in self.pieceswhite:
            piece.setPixmap(self.white)
        for piece in self.piecesblack:
            piece.setPixmap(self.black)
        for i in self.piecesblock:
            for j in range(len(i)):
                i[j].setPixmap(self.block)
                i[j].hide()
        self.pieceswhite[0].setGeometry(0 * 54 + 10, 6 * 54 + 10, self.white.width(), self.white.height())
        self.pieceswhite[1].setGeometry(3 * 54 + 10, 9 * 54 + 10, self.white.width(), self.white.height())
        self.pieceswhite[2].setGeometry(6 * 54 + 10, 9 * 54 + 10, self.white.width(), self.white.height())
        self.pieceswhite[3].setGeometry(9 * 54 + 10, 6 * 54 + 10, self.white.width(), self.white.height())
        self.piecesblack[0].setGeometry(0 * 54 + 10, 3 * 54 + 10, self.black.width(), self.black.height())
        self.piecesblack[1].setGeometry(3 * 54 + 10, 0 * 54 + 10, self.black.width(), self.black.height())
        self.piecesblack[2].setGeometry(6 * 54 + 10, 0 * 54 + 10, self.black.width(), self.black.height())
        self.piecesblack[3].setGeometry(9 * 54 + 10, 3 * 54 + 10, self.black.width(), self.black.height())
        self.setMouseTracking(True)
        # 定义模型
        self.array = None
        self.adjacency = None
        self.history = []

        self.show()

    def mouseMoveEvent(self, e):  # 棋子随鼠标移动
        self.mouse_point.move(e.x() - 18, e.y() - 18)

    def mousePressEvent(self, e):  # 玩家下棋
        global trainingsequence
        global trainingsequenceblack
        global trainingsequencewhite
        try:
            while 1:
                if check_list(trainingsequence, math.ceil(math.log2(self.chessboard.time))):
                    import json, os
                    # 创建目标文件夹
                    folder_name = "datasets-on-timeseries"
                    os.makedirs(folder_name, exist_ok=True)  # 如果文件夹不存在，则创建

                    # 初始化文件名和编号
                    base_filename = "dataset"
                    num = 0
                    filename = os.path.join(folder_name, f"{base_filename}[{num}].json")

                    # 检查文件是否已存在，直到找到一个可用的文件名
                    while os.path.exists(filename):
                        num += 1
                        filename = os.path.join(folder_name, f"{base_filename}[{num}].json")

                    # 保存为 JSON 文件
                    with open(filename, 'w') as json_file:
                        json.dump(trainingsequence, json_file)

                    print(f"数据已保存为 {filename}")
                    trainingsequence = []
                if check_list1(trainingsequenceblack, math.ceil(math.log2(self.chessboard.time))):
                    import json, os
                    # 创建目标文件夹
                    folder_name = "datasets-on-timeseries"
                    os.makedirs(folder_name, exist_ok=True)  # 如果文件夹不存在，则创建

                    # 初始化文件名和编号
                    base_filename1 = "dataset-black"
                    num = 0
                    filename = os.path.join(folder_name, f"{base_filename1}[{num}].json")

                    # 检查文件是否已存在，直到找到一个可用的文件名
                    while os.path.exists(filename):
                        num += 1
                        filename = os.path.join(folder_name, f"{base_filename1}[{num}].json")

                    # 保存为 JSON 文件
                    with open(filename, 'w') as json_file:
                        json.dump(trainingsequenceblack, json_file)

                    print(f"数据已保存为 {filename}")
                    trainingsequenceblack = []
                    print(trainingsequenceblack)
                if check_list1(trainingsequencewhite, math.ceil(math.log2(self.chessboard.time))):
                    import json, os
                    # 创建目标文件夹
                    folder_name = "datasets-on-timeseries"
                    os.makedirs(folder_name, exist_ok=True)  # 如果文件夹不存在，则创建

                    # 初始化文件名和编号
                    base_filename1 = "dataset-white"
                    num = 0
                    filename = os.path.join(folder_name, f"{base_filename1}[{num}].json")

                    # 检查文件是否已存在，直到找到一个可用的文件名
                    while os.path.exists(filename):
                        num += 1
                        filename = os.path.join(folder_name, f"{base_filename1}[{num}].json")

                    # 保存为 JSON 文件
                    with open(filename, 'w') as json_file:
                        json.dump(trainingsequencewhite, json_file)

                    print(f"数据已保存为 {filename}")
                    trainingsequencewhite = []
                    print(trainingsequencewhite)
                if self.turns % 2 == 1:
                    model1.load_state_dict(torch.load('autoencoder_model1.pth'))
                    model2.load_state_dict(torch.load('autoencoder_model2.pth'))
                    self.chessboard.autoencoder1 = model1
                    self.chessboard.autoencoder2 = model2
                    self.chessboard.autoencoder1.weight.to("cpu")
                    self.chessboard.autoencoder2.weight.to("cpu")
                    self.chessboard.autoencoder1.to("cpu")
                    self.chessboard.autoencoder2.to("cpu")
                    self.chessboard.blankpos = self.chessboard.blankpos - 1
                    # AI's move
                    if self.AI == WHITE:
                        self.AI = BLACK
                        mode = 0
                    else:
                        self.AI = WHITE
                        mode = 1
                    del_children(self.chessboard.nodeblack)
                    del_children(self.chessboard.nodewhite)
                    self.boardforcal = copy.deepcopy(self.chessboard)

                    # 这里是有的
                    feasible = []
                    feasible = self.boardforcal.searchchildren(mode, feasible)
                    if len(feasible) == 0:
                        self.gameover(3 - self.AI)
                    # searching step
                    limittime = 10
                    starttime = time.time()
                    while 1:
                        length = 0
                        print("start")
                        while feasible != -1:
                            feasible, length = self.boardforcal.nextnode_searching(self.turns, feasible, length,
                                                                                   GRPO_MCTS=self.GRPO_MCTS)
                            try:
                                print(len(feasible))
                            except:
                                pass
                        random.seed(int(time.time()))
                        # 选择两个点进行
                        if mode == 0:
                            temp = copy.deepcopy(self.boardforcal.nodeblack)
                        else:
                            temp = copy.deepcopy(self.boardforcal.nodewhite)
                        for node in temp:
                            node.update_node_values(node, 0)
                        for node in temp:
                            node.update_node_values2(node)
                        temp1 = []
                        for i in range(len(temp) - 1, -1, -1):
                            if temp[i].children:
                                temp1.append(temp[i])
                        temp = temp1
                        print(temp)
                        if len(temp) == 1:
                            temp.append(temp[0])

                        origin = copy.copy(temp)
                        print("---------searching is done---------")
                        print("---------selecting is initiating---------")
                        newnode = None
                        key = 0
                        try:
                            while 1:
                                newnode, num = self.boardforcal.geneticinit(0.7, temp, mode, origin)
                                if num != 1:
                                    break
                                else:
                                    continue
                            if time.time() - starttime > limittime:
                                break
                            self.update(newnode)
                            feasible = []
                            self.boardforcal.totaltime = 1
                            feasible = self.boardforcal.searchchildren(mode, feasible)
                        except Exception as e:
                            print(e)

                    self.chessboard.nodeblack = self.boardforcal.nodeblack
                    self.chessboard.nodewhite = self.boardforcal.nodewhite
                    self.chessboard.totaltime = self.boardforcal.totaltime
                    # todo 遗传算法之后对节点值进行再选择
                    adjacency = [
                        torch.zeros(self.boardforcal.totaltime, self.boardforcal.totaltime, dtype=torch.float32)]
                    array = [torch.zeros(self.boardforcal.totaltime, 7, dtype=torch.float32)]
                    newlist = list()
                    newlist += [0] * (self.boardforcal.totaltime)
                    self.arrayautodecoder1 = [torch.zeros(self.boardforcal.totaltime, 6, dtype=torch.float32)]
                    self.arrayautodecoder2 = [torch.zeros(self.boardforcal.totaltime, 6, dtype=torch.float32)]
                    # update the position of chessboard
                    self.turns = self.turns + 1
                    if mode == 0:
                        chess = self.chessboard.nodeblack
                    elif mode == 1:
                        chess = self.chessboard.nodewhite
                    dataconstruct(chess, adjacency, array, self.turns, self.arrayautodecoder1, self.arrayautodecoder2,
                                  newlist)
                    self.chessboard.autoencoder1.to(DEVICE)
                    self.chessboard.autoencoder2.to(DEVICE)
                    self.chessboard.autoencoder1.weight.to(DEVICE)
                    self.chessboard.autoencoder2.weight.to(DEVICE)
                    self.chessboard.autoencoder1.train()
                    self.chessboard.autoencoder2.train()
                    model1.weight = torch.load('weight1.pth')
                    model2.weight = torch.load('weight2.pth')
                    a = self.arrayautodecoder1[0][:, :-1]
                    b = self.arrayautodecoder1[0][:, -1].unsqueeze(1)
                    torch.save(a, "./train/in1.pth")
                    torch.save(b, "./train/out1.pth")
                    # for epoch in range(num_epochs):
                    #     optimizer1.zero_grad()
                    #     output = model1(a.to("cpu"))
                    #     loss = criterion1(output.to("cpu"), b.to("cpu"))
                    #     print(output)
                    #     print(self.chessboard.autoencoder1.weight)
                    #     print(a)
                    #     print(b)
                    #     loss.backward()
                    #     optimizer1.step()
                    #     print(loss)
                    torch.save(self.arrayautodecoder2[0][:, :-1], "./train/in2.pth")
                    torch.save(self.arrayautodecoder2[0][:, -1].unsqueeze(1), "./train/out2.pth")

                    # for epoch in range(num_epochs):
                    #     optimizer2.zero_grad()
                    #     output = model2(self.arrayautodecoder2[0][:, :-1].to("cpu"))
                    #     loss = criterion1(output.to("cpu"), self.arrayautodecoder2[0][:, -1].unsqueeze(1).to("cpu"))
                    #     print(output)
                    #     print(self.arrayautodecoder2[0][:, -1].unsqueeze(1))
                    #     loss.backward()
                    #     optimizer2.step()
                    #     print(loss)
                    # torch.save(model1.state_dict(), "autoencoder_model1.pth")
                    # torch.save(model1.weight, 'weight1.pth')
                    # torch.save(model2.state_dict(), "autoencoder_model2.pth")
                    # torch.save(model2.weight.weight, 'weight2.pth')
                    L = array[0][:, -1]
                    # A = adjacency[0]
                    # N = array[0][:, :-2]
                    torch.save(array[0][:, -1], "./train/L.pth")
                    torch.save(adjacency[0], "./train/A.pth")
                    torch.save(array[0][:, :-2], "./train/N.pth")
                    import os
                    os.system('python init.py')
                    # for epoch in range(num_epochs):
                    #     optimizer.zero_grad()
                    #     output = model(N.to(DEVICE), A.to(DEVICE))
                    #     print(output)
                    #     loss = criterion3(output.to(DEVICE)[:, 0], L.to(DEVICE))
                    #     print(L)
                    #     print(loss)
                    #     loss.backward()
                    #     optimizer.step()
                    model.autoencoder1 = model1.to(DEVICE)
                    model.autoencoder2 = model2.to(DEVICE)
                    A = adjacency[0]
                    N = array[0][:, :-2]
                    output = model(N.to(DEVICE), A.to(DEVICE))
                    print(output, L)
                    flattened_tensor = torch.abs(output - L.unsqueeze(1)).view(-1)
                    mask = (flattened_tensor[1:] != 0.5).tolist()
                    newlist = [newlist[1:][i] for i in range(len(mask)) if mask[i] == True]
                    flattened_tensor = torch.tensor(
                        [flattened_tensor[1:][i] for i in range(len(mask)) if mask[i] == True])
                    max_value, max_index = torch.max(flattened_tensor, dim=0)
                    anothernode = newlist[max_index.item()]

                    # 确定目标节点
                    # try:
                    #     newnode1 = random.choices([anothernode, newnode], [anothernode.obj, newnode.obj])[0]
                    # except:
                    #     newnode1 = anothernode
                    newnode.projection = projection1(newnode)
                    anothernode.projection = projection1(anothernode)
                    print(newnode)
                    print(anothernode)

                    def projection2(chessboard):
                        blacknode = copy.deepcopy(chessboard.nodeblack)
                        whitenode = copy.deepcopy(chessboard.nodewhite)
                        sequences = []
                        classification = []
                        # 造8个
                        for i in range(4):
                            x, y = projection(blacknode[i].pos[0], blacknode[i].pos[1])
                            x, y = add_noise(x, y)
                            r = math.sqrt(x ** 2 + y ** 2)
                            sequences.append(r)
                            classification.append(0)
                            x, y = projection(whitenode[i].pos[0], whitenode[i].pos[1])
                            x, y = add_noise(x, y)
                            r = math.sqrt(x ** 2 + y ** 2)
                            sequences.append(r)
                            classification.append(1)
                        # 排序
                        sequences, classification = new_sort(sequences, classification)
                        return sequences

                    board = projection2(self.chessboard)
                    print(board)
                    # 使用GRPO进行智能决策
                    newnode1, decision_info = integrate_grpo_decision(
                        original_newnode=newnode,
                        original_anothernode=anothernode,
                        chessboard=board,
                        tcn_output=output,  # 你的TCN输出
                        turns=self.turns
                    )
                    print(3)
                    newnode1.projection = projection1(newnode1)
                    self.grpo_trainer.record_selection(
                        newnode1.feasible_nodes, newnode1.feasible, newnode1.index
                    )
                    print(5)
                    self.grpo_trainer.train_with_reward(newnode1.obj)
                    print(6)
                    # 这里需要将两个node的projection计算出来

                    # call update (传 mapper)
                    mapper, amap = action.build_node_to_index_from_candidates(newnode1.feasible_nodes)

                    print(4)
                    self.last_candidates = newnode1
                    self.last_action_map = amap

                    print(self.history)
                    if self.history:
                        decision_info = self.update_grpo_after_move(
                            newnode=newnode1,  # 你的 newnode (Node instance)
                            all_candidates_next=feasible,
                            node_to_action_index=mapper,
                            K=6,
                            projection_func=projection1,
                            state_weight=1.0, policy_weight=1.0, value_weight=0.5,
                            exploration_gain=1.0, min_c=0.1, max_c=5.0,
                            device=DEVICE
                        )

                    # 确定主路径 todo
                    recycle = new_algorithm(temp, newnode1, 1 - newnode1.mode)
                    cur = newnode1
                    temp = []
                    while cur.parent:
                        temp.append(projection1(cur))
                        cur = getattr(cur, "parent", None)
                    print("'''")
                    print(temp)
                    print("'''")
                    self.history.append(temp)
                    from chessboard import minlimit, maxlimit
                    print("循环系数为：", recycle)
                    if self.chessboard.time * recycle < minlimit:
                        self.chessboard.time = minlimit
                    elif self.chessboard.time * recycle > maxlimit:
                        self.chessboard.time = maxlimit
                    else:
                        self.chessboard.time = self.chessboard.time * recycle
                    print("下一次循环次数：", self.chessboard.time)
                    self.update1(newnode1, 0)

                    #####
                    self.chessboard.nodeblack = self.boardforcal.nodeblack
                    self.chessboard.nodewhite = self.boardforcal.nodewhite
                    self.chessboard.totaltime = self.boardforcal.totaltime
                    print("------totaltime------")
                    try:
                        print(self.chessboard.totaltime)
                        del self.boardforcal
                        self.boardforcal = None

                        for node in self.chessboard.nodeblack:
                            node.board = self.chessboard.board()
                        for node in self.chessboard.nodewhite:
                            node.board = self.chessboard.board()
                        self.chessboard.totaltime = 1
                        print("end")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print("报错：")
                        print(e)

                else:
                    from chessboard import minlimit, maxlimit
                    temptime = self.chessboard.time
                    self.chessboard.time = minlimit
                    model1.load_state_dict(torch.load('autoencoder_model1.pth'))
                    model2.load_state_dict(torch.load('autoencoder_model2.pth'))
                    self.chessboard.autoencoder1 = model1
                    self.chessboard.autoencoder2 = model2
                    self.chessboard.autoencoder1.weight.to("cpu")
                    self.chessboard.autoencoder2.weight.to("cpu")
                    self.chessboard.autoencoder1.to("cpu")
                    self.chessboard.autoencoder2.to("cpu")
                    self.chessboard.blankpos = self.chessboard.blankpos - 1
                    # AI's move
                    if self.AI == WHITE:
                        self.AI = BLACK
                        mode = 0
                    else:
                        self.AI = WHITE
                        mode = 1
                    del_children(self.chessboard.nodeblack)
                    del_children(self.chessboard.nodewhite)
                    self.boardforcal = copy.deepcopy(self.chessboard)
                    feasible = []
                    feasible = self.boardforcal.searchchildren(mode, feasible)
                    if len(feasible) == 0:
                        self.gameover(3 - self.AI)
                    # searching step
                    limittime = 10
                    starttime = time.time()
                    while 1:
                        length = 0
                        print("start")
                        while feasible != -1:
                            feasible, length = self.boardforcal.nextnode_searching(self.turns, feasible, length)
                            try:
                                print(len(feasible))
                            except:
                                pass
                        random.seed(int(time.time()))
                        # 选择两个点进行 todo
                        if mode == 0:
                            temp = copy.deepcopy(self.boardforcal.nodeblack)
                        else:
                            temp = copy.deepcopy(self.boardforcal.nodewhite)
                        for node in temp:
                            node.update_node_values(node, 0)
                        for node in temp:
                            node.update_node_values2(node)
                        temp1 = []
                        for i in range(len(temp) - 1, -1, -1):
                            if temp[i].children:
                                temp1.append(temp[i])
                        temp = temp1
                        print(temp)
                        if len(temp) == 1:
                            temp.append(temp[0])
                        break
                    self.chessboard.nodeblack = self.boardforcal.nodeblack
                    self.chessboard.nodewhite = self.boardforcal.nodewhite
                    self.chessboard.totaltime = self.boardforcal.totaltime
                    # todo 遗传算法之后对节点值进行再选择
                    adjacency = [
                        torch.zeros(self.boardforcal.totaltime, self.boardforcal.totaltime, dtype=torch.float32)]
                    array = [torch.zeros(self.boardforcal.totaltime, 7, dtype=torch.float32)]
                    newlist = list()
                    newlist += [0] * (self.boardforcal.totaltime)
                    self.arrayautodecoder1 = [torch.zeros(self.boardforcal.totaltime, 6, dtype=torch.float32)]
                    self.arrayautodecoder2 = [torch.zeros(self.boardforcal.totaltime, 6, dtype=torch.float32)]
                    # update the position of chessboard
                    self.turns = self.turns + 1
                    if mode == 0:
                        chess = self.chessboard.nodeblack
                    elif mode == 1:
                        chess = self.chessboard.nodewhite
                    dataconstruct(chess, adjacency, array, self.turns, self.arrayautodecoder1, self.arrayautodecoder2,
                                  newlist)
                    temp = []
                    for i in range(len(newlist)):
                        try:
                            temp.append(newlist[i].obj)
                        except:
                            temp.append(0)
                    newnode = random.choices(newlist, temp)[0]

                    cur = newnode
                    temp = []
                    while cur.parent:
                        temp.append(projection1(cur))
                        cur = getattr(cur, "parent", None)
                    self.history.append(temp)

                    self.update1(newnode, 1)

                    self.chessboard.nodeblack = self.boardforcal.nodeblack
                    self.chessboard.nodewhite = self.boardforcal.nodewhite
                    self.chessboard.totaltime = self.boardforcal.totaltime
                    #####
                    print("------totaltime------")
                    self.chessboard.time = temptime
                    try:
                        print(self.chessboard.totaltime)
                        del self.boardforcal
                        self.boardforcal = None
                        for node in self.chessboard.nodeblack:
                            node.board = self.chessboard.board()
                        for node in self.chessboard.nodewhite:
                            node.board = self.chessboard.board()
                        self.chessboard.totaltime = 1
                        print("end")
                    except Exception as e:
                        print("报错：")
                        print(e)

        except Exception as e:
            import traceback
            print(traceback.print_exc())
            # 创建目标文件夹
            folder_name = "datasets-on-timeseries"
            os.makedirs(folder_name, exist_ok=True)  # 如果文件夹不存在，则创建

            # 初始化文件名和编号
            base_filename = "dataset"
            num = 0
            filename = os.path.join(folder_name, f"{base_filename}[{num}].json")

            # 检查文件是否已存在，直到找到一个可用的文件名
            while os.path.exists(filename):
                num += 1
                filename = os.path.join(folder_name, f"{base_filename}[{num}].json")

            # 保存为 JSON 文件
            with open(filename, 'w') as json_file:
                json.dump(trainingsequence, json_file)

            print(f"数据已保存为 {filename}")

    def update(self, node):
        self.boardforcal.boardchange(node.board)
        self.boardforcal.nodeblack = node.black
        self.boardforcal.nodewhite = node.white
        self.boardforcal.nodeblock = node.block

    def update1(self, node, mode):
        temp = copy.deepcopy(node)
        list2 = []
        sequences = []
        classification = []
        if temp.parent is not None:
            while temp.parent.parent is not None:
                list1 = temp.black + temp.white
                sequence = []
                radius = []
                for n in range(len(list1)):
                    if n <= 3:
                        sequence.append(0)
                        x, y = projection(list1[n].pos[0], list1[n].pos[1])
                        x, y = add_noise(x, y)
                        r = math.sqrt(x ** 2 + y ** 2)
                        radius.append(r)
                    else:
                        sequence.append(1)
                        x, y = projection(list1[n].pos[0], list1[n].pos[1])
                        x, y = add_noise(x, y)
                        r = math.sqrt(x ** 2 + y ** 2)
                        radius.append(r)
                radius, sequence = new_sort(radius, sequence)
                list2.append(radius)
                temp = copy.deepcopy(temp.parent)
        if mode == 0:
            trainingsequence[:0] = [list2]
            for i in range(4):
                x, y = projection(node.black[i].pos[0], node.black[i].pos[1])
                x, y = add_noise(x, y)
                r = math.sqrt(x ** 2 + y ** 2)
                sequences.append(r)
                classification.append(0)
            for i in range(4):
                x, y = projection(node.white[i].pos[0], node.white[i].pos[1])
                x, y = add_noise(x, y)
                r = math.sqrt(x ** 2 + y ** 2)
                sequences.append(r)
                classification.append(1)
            trainingsequenceblack[:0] = [sequences]
        if mode == 1:
            for i in range(4):
                x, y = projection(node.black[i].pos[0], node.black[i].pos[1])
                x, y = add_noise(x, y)
                r = math.sqrt(x ** 2 + y ** 2)
                sequences.append(r)
                classification.append(0)
            for i in range(4):
                x, y = projection(node.white[i].pos[0], node.white[i].pos[1])
                x, y = add_noise(x, y)
                r = math.sqrt(x ** 2 + y ** 2)
                sequences.append(r)
                classification.append(1)
            trainingsequencewhite[:0] = [sequences]
        print(trainingsequence, trainingsequenceblack, trainingsequencewhite)
        block = diffrence(self.boardforcal.nodeblock, temp.block)
        print("new block:")
        print(block)
        self.lastx = temp.parent.pos[0]
        self.lasty = temp.parent.pos[1]
        print(self.lastx, self.lasty)
        print(temp.pos[0], temp.pos[1])
        # 更新AI的
        if self.AI == WHITE:
            self.draw2(temp.pos[0], temp.pos[1], 1)
            for node in self.boardforcal.nodewhite:
                if temp.parent.pos == node.pos:
                    node.pos = temp.pos
                    print("替换成功")
                    break
        elif self.AI == BLACK:
            self.draw2(temp.pos[0], temp.pos[1], 0)
            for node in self.boardforcal.nodeblack:
                if temp.parent.pos == node.pos:
                    node.pos = temp.pos
                    break
        for i in block:
            self.draw(i[0], i[1], 1)
        self.boardforcal.boardchange(node.board)
        return node

    def playermove(self, e):  # 落子
        if e.button() == Qt.LeftButton:
            if not self.click:
                i, j = int((e.x() - 10) / 54), int((e.y() - 10) / 54)  # 鼠标坐标   (i,j)
                if self.first == WHITE:
                    for k in self.chessboard.nodewhite:
                        k = k.pos
                        if i == int(k[0]) and j == int(k[1]):
                            # 标记选择点位置
                            self.cursorlabel.setGeometry(i * 54, j * 54, self.cursor.width(), self.cursor.height())
                            self.cursorlabel.show()
                            # 拿起棋子
                            self.mouse_point.setScaledContents(True)
                            self.mouse_point.setPixmap(self.white)  # 加载白棋
                            self.mouse_point.setGeometry(270, 270, PIECE, PIECE)
                            self.mouse_point.raise_()  # 鼠标始终在最上层
                            self.lastx = i
                            self.lasty = j
                            for mm in range(4):
                                if int((self.pieceswhite[mm].x() - 10) / 54) == i and int(
                                        (self.pieceswhite[mm].y() - 10) / 54) == j:
                                    self.pieceswhite[mm].hide()
                                    self.chessboard.x = int((self.pieceswhite[mm].x() - 10) / 54)
                                    self.chessboard.y = int((self.pieceswhite[mm].y() - 10) / 54)
                                    self.click = 1 - self.click
                                    return 0
                    self.click = 0
                    return 2

                elif self.first == BLACK:
                    for k in self.chessboard.nodeblack:
                        k = k.pos
                        if i == int(k[0]) and j == int(k[1]):
                            # 标记选择点位置
                            self.cursorlabel.setGeometry(i * 54, j * 54, self.cursor.width(), self.cursor.height())
                            self.cursorlabel.show()
                            # 拿起棋子
                            self.mouse_point.setScaledContents(True)
                            self.mouse_point.setPixmap(self.black)  # 加载黑棋
                            self.mouse_point.setGeometry(i * 54, j * 54, PIECE, PIECE)
                            self.mouse_point.raise_()  # 鼠标始终在最上层
                            self.lastx = i
                            self.lasty = j
                            for mm in range(4):
                                if int((self.piecesblack[mm].x() - 10) / 54) == i and int(
                                        (self.piecesblack[mm].y() - 10) / 54) == j:
                                    self.piecesblack[mm].hide()
                                    self.chessboard.x = int((self.pieceswhite[mm].x() - 10) / 54)
                                    self.chessboard.y = int((self.pieceswhite[mm].y() - 10) / 54)
                                    self.click = 1 - self.click
                                    return 0
                    self.click = 0
                    return 2
            else:
                if e.button() == Qt.LeftButton and self.my_turn == True:
                    i, j = int(e.x() / 54), int(e.y() / 54)  # 鼠标坐标
                    if not (self.chessboard.get_xy_on_logic_state(i, j) and self.chessboard.judge_xy_on_logic_state(i,
                                                                                                                    j,
                                                                                                                    self.lastx,
                                                                                                                    self.lasty)):
                        return 0
                    self.draw(i, j)  # 移动
                    self.mouse_point.clear()
                    self.cursorlabel.hide()
                    self.click = 1 - self.click
                    return 1

    def playerput(self, e):
        if e.button() == Qt.LeftButton:
            i, j = int(e.x() / 54), int(e.y() / 54)  # 鼠标坐标
            if self.chessboard.get_xy_on_logic_state(i, j):
                self.draw(i, j, 1)  # 做一个障碍物
                return 1
        return 0

    def draw(self, i, j, mode=0):
        x = int((i * 54 + 10))
        y = int((j * 54 + 10))
        if mode == 1:
            self.chessboard.draw_xy(i, j, 2)
            self.piecesblock[i][j].show()  # 展示障碍物
            self.piecesblock[i][j].setGeometry(x - 10, y - 10, self.block.width(), self.block.height())
            self.blocknum = self.blocknum + 1
            return
        elif self.first == BLACK:
            for p in range(4):
                if self.lastx == int((self.piecesblack[p].x() - 10) / 54) and self.lasty == int(
                        (self.piecesblack[p].y() - 10) / 54):
                    self.piecesblack[p].setGeometry(x, y, self.black.width(), self.black.height())  # 改界面棋子位置
                    self.piecesblack[p].show()
                    self.chessboard.draw_xy(i, j, 0, self.lastx, self.lasty)  # 改系统棋子位置
                    break

        elif self.first == WHITE:
            for p in range(4):
                if self.lastx == int((self.piecesblack[p].x() - 10) / 54) and self.lasty == int(
                        (self.piecesblack[p].y() - 10) / 54):
                    self.pieceswhite[p].setGeometry(x, y, self.white.width(), self.white.height())  # 改界面棋子位置
                    self.pieceswhite[p].show()
                    self.chessboard.draw_xy(i, j, 1, self.lastx, self.lasty)  # draw_xy要改 改系统棋子位置
                    break

    def draw2(self, i, j, mode):
        x = int((i * 54 + 10))
        y = int((j * 54 + 10))

        if mode == 0:
            for p in range(4):
                if self.lastx == int((self.piecesblack[p].x() - 10) / 54) and self.lasty == int(
                        (self.piecesblack[p].y() - 10) / 54):
                    self.piecesblack[p].setGeometry(x, y, self.black.width(), self.black.height())  # 改界面棋子位置
                    self.piecesblack[p].show()
                    self.chessboard.draw_xy(i, j, 0, self.lastx, self.lasty)  # 改系统棋子位置
                    break

        elif mode == 1:
            for p in range(4):
                if self.lastx == int((self.pieceswhite[p].x() - 10) / 54) and self.lasty == int(
                        (self.pieceswhite[p].y() - 10) / 54):
                    self.pieceswhite[p].setGeometry(x, y, self.white.width(), self.white.height())  # 改界面棋子位置
                    self.pieceswhite[p].show()
                    self.chessboard.draw_xy(i, j, 1, self.lastx, self.lasty)  # draw_xy要改 改系统棋子位置
                    break

    def gameover(self, winner):
        if winner == BLACK:
            print("BLACK")
            self.sound_win.play()
            reply = QMessageBox.question(self, 'You Win!', 'Continue?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            print("WHITE")
            self.sound_defeated.play()
            reply = QMessageBox.question(self, 'You Lost!', 'Continue?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:  # 复位
            self.piece_now = BLACK
            self.mouse_point.setPixmap(self.black)
            self.step = 0
            for piece in self.pieces:
                piece.clear()
            self.chessboard.reset()
            self.update()
        else:
            self.close()

    def update_grpo_after_move(
            self,
            newnode,
            all_candidates_next=None,  # feasible
            K=6,  # history length: K-1 historical projections + current
            projection_func=projection1,  # projection1
            node_to_action_index=None,  # optional override
            state_weight=1.0,
            policy_weight=1.0,
            value_weight=0.5,
            exploration_gain=1.0,
            min_c=0.1, max_c=5.0,
            device=DEVICE
    ):
        """
        在每回合结束调用，使用 self.history（投影序列）和 newnode 做 TCN+PV 预测，
        计算预测 vs 实际的差异，并把调整写回 GRPO（grpo.last_adjusted_c, grpo.next_round_biases）。
        Returns: decision_info dict (tcn_pred, policy_probs, value_pred, temporal_pred,
                 state_error, policy_error, value_error, combined_error, adjusted_c, candidate_biases)
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            grpo = getattr(self, "GRPO_MCTS", None)
            tcn_model = getattr(self, "flexible_tcn_model", None)
            pv_net = getattr(self, "policy_value_net", None)

            # ----- projection fallback -----
            proj = projection_func or getattr(self, "projection_func", None)

            # ----- inline binary node->action mapper (origin,dest only) -----
            if node_to_action_index is None:
                # 默认 binary mapper
                def _pos_to_index_basic(pos, board_size=10):
                    if pos is None:
                        return None
                    if isinstance(pos, int):
                        return int(pos)
                    if hasattr(pos, "__len__"):
                        x = int(pos[0])
                        y = int(pos[1])
                        return x * board_size + y
                    raise ValueError(f"Unsupported pos format: {pos}")

                def node_to_action_index_binary_local(feasible_node, board_size=10, action_space=None,
                                                      allow_modulo=True):
                    origin = None
                    dest = None
                    try:
                        if isinstance(feasible_node, (list, tuple)):
                            if len(feasible_node) >= 2:
                                origin = feasible_node[0]
                                dest = feasible_node[1]
                        else:
                            origin = getattr(feasible_node, "origin", None) or getattr(feasible_node, "from", None)
                            dest = getattr(feasible_node, "dest", None) or getattr(feasible_node, "to", None)
                    except Exception:
                        pass
                    if origin is None or dest is None:
                        try:
                            if isinstance(feasible_node, (list, tuple)) and len(feasible_node) > 4:
                                prev = feasible_node[4]
                                origin = origin or getattr(prev, "origin", None) or getattr(prev, "from", None)
                                dest = dest or getattr(prev, "dest", None) or getattr(prev, "to", None)
                        except Exception:
                            pass
                    if origin is None or dest is None:
                        return None
                    oidx = _pos_to_index_basic(origin, board_size)
                    didx = _pos_to_index_basic(dest, board_size)
                    if oidx is None or didx is None:
                        return None
                    cells = board_size * board_size
                    idx = int(oidx * cells + didx)
                    if action_space is not None:
                        max_range = cells * cells
                        if action_space < max_range:
                            if allow_modulo:
                                return int(idx % action_space)
                            else:
                                return None
                    return int(idx)

                default_action_space = getattr(self, "policy_action_space", None) or getattr(self, "action_space",
                                                                                             None) or 2700
                action_mapper = lambda fe: node_to_action_index_binary_local(fe, board_size=10,
                                                                             action_space=default_action_space,
                                                                             allow_modulo=True)
            else:
                # 如果传入的是 dict，把它包装成 mapper；如果是 callable，直接用
                if isinstance(node_to_action_index, dict):
                    map_dict = node_to_action_index

                    def action_mapper(fe):
                        # 先尝试 canonical key lookup（支持 feasible list 格式）
                        try:
                            key = None
                            if isinstance(fe, (list, tuple)):
                                # canonical: (orig,dest,mode)
                                orig = fe[0] if len(fe) > 0 else None
                                dest = fe[1] if len(fe) > 1 else None
                                mode = fe[2] if len(fe) > 2 else None
                                key = (tuple(orig) if hasattr(orig, "__len__") and not isinstance(orig, int) else orig,
                                       tuple(dest) if hasattr(dest, "__len__") and not isinstance(dest, int) else dest,
                                       mode)
                            if key is not None and key in map_dict:
                                return map_dict[key]
                        except Exception:
                            pass
                        # fallback to direct dict lookup (maybe keys are node objects)
                        return map_dict.get(fe, None)
                elif callable(node_to_action_index):
                    action_mapper = node_to_action_index
                    print("here")
                else:
                    # unknown type -> fallback to None-mapper
                    def action_mapper(fe):
                        return None

            decision_info = {}

            # --------- 1) assemble history projections ----------
            history = getattr(self, "history", [])  # todo history更新有问题
            hist_proj = []
            print("这里是新的")
            print(history)
            for h in history[-(K - 1):]:
                if isinstance(h, torch.Tensor):
                    hist_proj.append(h.detach().cpu().numpy().astype(np.float32))
                else:
                    hist_proj.append(np.array(h, dtype=np.float32))

            def preprocess_all(seq_list, model):
                """
                对 seq_list 中的相邻 pair 依次运行 preprocess 逻辑，并原地修改 seq_list。
                seq_list: 三维列表，形状类似 [ [D维向量...], [D维向量...], ... ]
                model: 你的 tcn_model
                """
                for i in range(len(seq_list) - 1):
                    temp1_list = seq_list[i]
                    temp2_list = seq_list[i + 1]

                    # 转成 Tensor
                    t1 = torch.tensor(temp1_list)
                    t2 = torch.tensor(temp2_list)

                    # 上一层输入（这里我假设用前面所有的历史合成）
                    all_prev = torch.tensor([item for sub in seq_list[:i + 1] for item in sub]).unsqueeze(0)

                    # 原来的填充逻辑
                    if t1.size(0) > t2.size(0) + 1:
                        start = t2.size(0)
                        end = t1.size(0) - 1
                        with torch.no_grad():
                            for j in range(start, end):
                                x1 = t1[j].unsqueeze(0)
                                x2 = t1[j + 1].unsqueeze(0)
                                x3 = t2[j - 1].unsqueeze(0)
                                x4 = all_prev
                                output1 = model(x1, x2, j, x4, x3)
                                t2 = torch.cat([t2, output1.squeeze(0).detach()], dim=0)

                        # 原地更新
                        seq_list[i + 1] = [t2[k] for k in range(t2.size(0))]

                    elif len(temp1_list) < len(temp2_list) + 1:
                        seq_list[i + 1] = seq_list[i + 1][: len(temp1_list) - 1]

                return seq_list  # 可返回也可不返回

            hist_proj = preprocess_all(hist_proj, tcn_model)
            print(hist_proj)
            # --------- 2) TCN predict ----------
            input_prev_np = np.stack(
                [h.detach().cpu().numpy() if isinstance(h, torch.Tensor) else np.array(h, dtype=np.float32)
                 for h in hist_proj],
                axis=0
            )
            input_prev = torch.tensor(input_prev_np, dtype=torch.float32).unsqueeze(0).to(device)
            last_state = hist_proj[-1]
            # last_state = hist_proj[-1] if len(hist_proj) > 0 else np.zeros(D, dtype=np.float32)
            last_state_t = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(device)

            if tcn_model is not None:
                tcn_model.eval()
                with torch.no_grad():
                    in1 = last_state_t
                    in2 = last_state_t
                    try:
                        num_arg = min(len(hist_proj) - 1,
                                      getattr(tcn_model, "maxlimit", 1) - 1 if hasattr(tcn_model, "maxlimit") else 0)
                        tcn_out = tcn_model(in1, in2, num_arg, input_prev, None)
                        if isinstance(tcn_out, torch.Tensor):
                            if tcn_out.dim() == 1:
                                tcn_pred = tcn_out.unsqueeze(0).cpu().numpy()
                            else:
                                tcn_pred = tcn_out.detach().cpu().numpy()
                                if tcn_pred.shape[0] == 1 and tcn_pred.ndim > 1:
                                    tcn_pred = tcn_pred.reshape(1, -1)
                        else:
                            tcn_pred = np.array(tcn_out, dtype=np.float32).reshape(1, -1)
                    except Exception:
                        tcn_pred = last_state_t.detach().cpu().numpy().reshape(1, -1)
            else:
                tcn_pred = last_state_t.detach().cpu().numpy().reshape(1, -1)

            decision_info['tcn_pred'] = tcn_pred.squeeze(0)

            # --------- 3) Policy-Value predict ----------
            if pv_net is None:
                raise RuntimeError("policy_value_net not found on self; required for prediction.")

            pv_net.eval()
            with torch.no_grad():
                cur_state_t = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(device)
                tcn_pred_t = torch.tensor(tcn_pred.squeeze(0), dtype=torch.float32).unsqueeze(0).to(device)
                current_step = torch.tensor([len(history)], dtype=torch.long).to(device)
                expected_step = torch.tensor([len(history) + 1], dtype=torch.long).to(device)
                print(cur_state_t, tcn_pred_t, current_step, expected_step)
                policy_logits, value_pred, temporal_pred, _att = pv_net(cur_state_t, tcn_pred_t, current_step,
                                                                        expected_step)

                # 确保是 tensor 且形状为 [B, C]
                if not torch.is_tensor(policy_logits):
                    policy_logits = torch.tensor(np.array(policy_logits), dtype=torch.float32)

                if policy_logits.dim() == 1:
                    policy_logits = policy_logits.unsqueeze(0)  # [1, C]
                # 现在做 softmax（按最后一维）
                policy_probs_tensor = torch.softmax(policy_logits, dim=-1)  # [B, C]
                policy_probs = policy_probs_tensor.cpu().numpy()
                if policy_probs.shape[0] == 1:
                    policy_probs = policy_probs[0]  # 变为 1D array [C]

                # ---------- value ----------
                if not torch.is_tensor(value_pred):
                    value_pred = torch.tensor(np.array(value_pred), dtype=torch.float32)
                value_np = value_pred.detach().cpu().numpy()
                value_v = float(value_np.squeeze()) if value_np.size == 1 else value_np

                # ---------- temporal ----------
                if not torch.is_tensor(temporal_pred):
                    temporal_pred = torch.tensor(np.array(temporal_pred), dtype=torch.float32)
                temporal_v = temporal_pred.detach().cpu().numpy()

            decision_info['policy_probs'] = policy_probs
            decision_info['value_pred'] = value_v
            decision_info['temporal_pred'] = temporal_v

            # --------- 4) actual proj for newnode and compute errors ----------
            actual_proj = None
            try:
                board_candidate = None
                if isinstance(newnode, (list, tuple)) and len(newnode) > 4:
                    prev = newnode[4]
                    board_candidate = getattr(prev, "board", None) or getattr(prev, "state", None)
                if board_candidate is None:
                    board_candidate = getattr(self, "chessboard", None)
                if board_candidate is not None:
                    actual_proj = proj(board_candidate)
                else:
                    actual_proj = proj(newnode)
            except Exception:
                actual_proj = proj(newnode)

            actual_proj = (
                actual_proj.detach().cpu().numpy() if isinstance(actual_proj, torch.Tensor) else np.array(actual_proj,
                                                                                                          dtype=np.float32))
            predicted_proj = np.array(tcn_pred.squeeze(0), dtype=np.float32)

            state_err = float(np.linalg.norm(predicted_proj - actual_proj)) / (np.linalg.norm(predicted_proj) + 1e-6)
            decision_info['state_error'] = state_err

            # --------- 5) policy error for actual newnode action (using binary mapper) ----------
            try:
                action_idx = action_mapper(newnode)
            except Exception:
                # fallback: use binary mapper without modulo
                action_idx = node_to_action_index_binary_local(newnode, board_size=10, action_space=None,
                                                               allow_modulo=True)
            policy_prob_for_actual = 0.0
            if action_idx is not None and action_idx < len(policy_probs):
                policy_prob_for_actual = float(policy_probs[action_idx])
            else:
                policy_prob_for_actual = 1.0 / max(1, len(policy_probs))
            policy_err = 1.0 - policy_prob_for_actual
            decision_info['policy_prob_for_actual'] = policy_prob_for_actual
            decision_info['policy_error'] = policy_err

            # --------- 6) value error ----------
            node_obj_val = float(newnode[3]) if len(newnode) > 3 else 0.0

            node_obj_norm = math.tanh(node_obj_val / (abs(node_obj_val) + 1e-6)) if abs(node_obj_val) > 0 else 0.0
            value_err = float(abs(value_v - node_obj_norm))
            decision_info['node_obj_norm'] = node_obj_norm
            decision_info['value_error'] = value_err

            # --------- 7) combined error and normalized score ----------
            combined = state_weight * state_err + policy_weight * policy_err + value_weight * value_err
            combined_norm = 1.0 / (1.0 + math.exp(- (combined - 0.5)))
            combined_norm = float(max(0.0, min(1.0, combined_norm)))
            decision_info['combined_error_raw'] = combined
            decision_info['combined_error'] = combined_norm

            # --------- 8) adjust exploration constant using grpo then scale by combined error ----------
            try:
                group_state = grpo.analyze_group_state(all_candidates_next if all_candidates_next is not None else [])
                base_c = getattr(grpo, "last_adjusted_c", getattr(grpo, "default_c", getattr(self, "c", 1.0)))
                grpo_c = grpo.adjust_exploration_constant(
                    base_c=base_c,
                    group_diversity=(
                        float(group_state[6].item()) if torch.is_tensor(group_state) and group_state.size(0) > 6 else (
                            float(group_state[6]) if len(group_state) > 6 else 0.0)),
                    consensus_level=1.0 - (
                        float(group_state[7].item()) if torch.is_tensor(group_state) and group_state.size(0) > 7 else (
                            float(group_state[7]) if len(group_state) > 7 else 0.0)),
                    search_progress=(getattr(self, "turns", 1) / (getattr(self, "turns", 1) + 1.0))
                )
            except Exception:
                grpo_c = getattr(grpo, "default_c", getattr(self, "c", 1.0))

            final_c = grpo_c * (1.0 + exploration_gain * combined_norm)
            final_c = max(min_c, min(max_c, final_c))
            decision_info['adjusted_c'] = final_c
            try:
                grpo.last_adjusted_c = final_c
            except Exception:
                setattr(grpo, "last_adjusted_c", final_c)

            # --------- 9) compute per-candidate biases ----------
            candidate_biases = {}
            eps = 1e-9
            alpha = 5.0
            lamb = 0.5
            # policy_probs length
            pol_len = len(policy_probs)
            for idx, cand in enumerate(all_candidates_next if all_candidates_next is not None else []):
                try:
                    cand_board = None
                    if isinstance(cand, (list, tuple)) and len(cand) > 4:
                        cand_board = getattr(cand[4], "board", None) or getattr(cand[4], "state", None)
                    if cand_board is None:
                        cand_proj = proj(cand)
                    else:
                        cand_proj = proj(cand_board)
                    cand_proj = (
                        cand_proj.detach().cpu().numpy() if isinstance(cand_proj, torch.Tensor) else np.array(cand_proj,
                                                                                                              dtype=np.float32))
                except Exception:
                    cand_proj = np.zeros_like(predicted_proj)

                dist = np.linalg.norm(cand_proj - predicted_proj)
                state_sim = math.exp(-alpha * dist)

                try:
                    aidx = action_mapper(cand)
                    pscore = float(policy_probs[aidx]) if (aidx is not None and aidx < pol_len) else (
                                1.0 / max(1, pol_len))
                except Exception:
                    pscore = 1.0 / max(1, pol_len)

                bias = (math.log(pscore + eps) * policy_weight) + (lamb * state_sim)
                candidate_biases[idx] = float(bias)

            decision_info['candidate_biases'] = candidate_biases
            grpo.next_round_biases = candidate_biases

            # --------- 10) push experience for learning ----------
            if hasattr(grpo, "experience_buffer"):
                # prepare group_state and node_features for this experience
                group_state_tensor = grpo.analyze_group_state(
                    all_candidates_next if all_candidates_next is not None else [])
                try:
                    node_features_tensor = grpo.extract_node_features(
                        [newnode.parent.pos, newnode.pos, newnode.mode, newnode.obj, newnode.parent])
                except Exception:
                    node_features_tensor = torch.zeros(8, dtype=torch.float32)

                # 定义 reward：这里把 combined_norm 越小看作越好，所以 reward = 1 - combined_norm
                reward = float(1.0 - combined_norm)

                exp = {
                    "timestamp": getattr(grpo, "selection_count", 0),
                    "node_features": node_features_tensor.detach().cpu() if torch.is_tensor(
                        node_features_tensor) else node_features_tensor,
                    "group_state": group_state_tensor.detach().cpu() if torch.is_tensor(
                        group_state_tensor) else group_state_tensor,
                    "pred_tcn": predicted_proj,
                    "actual_proj": actual_proj,
                    "policy_probs": policy_probs,
                    "policy_prob_for_actual": policy_prob_for_actual,
                    "value_pred": value_v,
                    "node_obj_norm": node_obj_norm,
                    "state_error": state_err,
                    "policy_error": policy_err,
                    "value_error": value_err,
                    "combined_error": combined_norm,
                    "reward": reward
                }
                grpo.experience_buffer.append(exp)

                # --------- 额外：触发一次 GRPO 在线更新并把 loss 保存到文件 ----------
                try:
                    loss = None
                    # 如果 grpo 有 _update_network 方法，就调用它（它现在应该返回 loss 或 None）
                    if hasattr(grpo, "_update_network") and callable(getattr(grpo, "_update_network")):
                        loss = grpo._update_network()

                    # 如果获得了 loss（float），把它追加写入 grpo.base_dir/grpo_after_move.txt
                    if loss is not None:
                        import os
                        base_dir = getattr(grpo, "base_dir", "./train") or "./train"
                        log_path = os.path.join(base_dir, "grpo_after_move.txt")
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        try:
                            with open(log_path, "a", encoding="utf-8") as lf:
                                lf.write(f"{loss:.6f}\n")
                        except Exception as e:
                            print(f"[Warning] 无法写入 loss 日志文件: {e}")
                except Exception as e:
                    print(f"[Warning] 触发 GRPO 在线更新或写日志时出错: {e}")

        except:
            import traceback
            traceback.print_exc()
        return decision_info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    app.exec_()
