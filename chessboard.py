#!/usr/bin/env python
# -*- coding:utf-8 -*-
import copy
from time import sleep

import numpy as np

# ----------------------------------------------------------------------
# 定义棋子类型，输赢情况
# ----------------------------------------------------------------------
EMPTY = 0
BLACK = 1
WHITE = 2
Block = 3
rate = 0.7
minlimit = 20
maxlimit = 50

from AiTree import *
from multiprocessing import Process, RLock, Manager
import time
import pickle
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn as nn
import sys
import math
from GRPO_MCTS import *

sys.setrecursionlimit(3000)  # 修改递归深度为 3000
random.seed(time.time())  # 随机种子

DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


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


def softmax(list1):
    return np.exp(np.array(list1)) / sum(np.exp(np.array(list1)))


def update(onenode0, onenode1, onenode2, onenode3, mode, temp):  # [上个节点，放置位置，原位置]
    if mode == 0:
        nodepos = copy.deepcopy(onenode0.black)
        for count in range(len(onenode0.black)):
            if onenode0.black[count].pos[0] == onenode2[0] and onenode0.black[count].pos[1] == onenode2[1]:
                nodepos[count].pos = onenode1
                break
        onenode0.add_child(onenode1, -1, True, temp, onenode0.white, nodepos, onenode3, 1 - onenode0.max,
                           1 - mode)
    elif mode == 1:
        nodepos = copy.deepcopy(onenode0.white)
        for count in range(len(onenode0.white)):
            if onenode0.white[count].pos[0] == onenode2[0] and onenode0.white[count].pos[1] == onenode2[1]:
                nodepos[count].pos = onenode1
                break
        onenode0.add_child(onenode1, -1, True, temp, nodepos, onenode0.black, onenode3, 1 - onenode0.max,
                           1 - mode)


# ----------------------------------------------------------------------
# 定义棋盘类，绘制棋盘的形状，切换先后手，判断输赢等
# ----------------------------------------------------------------------
# nodeblack、nodewhite、nodeblock基础节点
class ChessBoard(object):
    totaltime = 1
    blankpos = 92

    def __init__(self, board=None, turn=0, paras=None, paras2=None):
        if not board:
            self.__board = [[EMPTY for n in range(10)] for m in range(10)]
            self.__board[0][3] = BLACK
            self.__board[3][0] = BLACK
            self.__board[6][0] = BLACK
            self.__board[9][3] = BLACK
            self.__board[0][6] = WHITE
            self.__board[3][9] = WHITE
            self.__board[6][9] = WHITE
            self.__board[9][6] = WHITE
        self.white = [[EMPTY for n in range(10)] for m in range(10)]
        self.black = [[EMPTY for n in range(10)] for m in range(10)]
        self.nodeblack = [Node(None, [], self.board(), [], [], [], -1, True) for i in range(4)]
        self.nodeblack[0].pos = [0, 3]
        self.nodeblack[1].pos = [3, 0]
        self.nodeblack[2].pos = [6, 0]
        self.nodeblack[3].pos = [9, 3]
        self.nodewhite = [Node(None, [], self.board(), [], [], [], -1, True) for i in range(4)]
        self.nodewhite[0].pos = [0, 6]
        self.nodewhite[1].pos = [3, 9]
        self.nodewhite[2].pos = [6, 9]
        self.nodewhite[3].pos = [9, 6]
        self.nodeblack[0].black = self.nodeblack
        self.nodeblack[1].black = self.nodeblack
        self.nodeblack[2].black = self.nodeblack
        self.nodeblack[3].black = self.nodeblack
        self.nodewhite[0].white = self.nodewhite
        self.nodewhite[1].white = self.nodewhite
        self.nodewhite[2].white = self.nodewhite
        self.nodewhite[3].white = self.nodewhite
        self.nodewhite[0].black = self.nodeblack
        self.nodewhite[1].black = self.nodeblack
        self.nodewhite[2].black = self.nodeblack
        self.nodewhite[3].black = self.nodeblack
        self.nodeblack[0].white = self.nodewhite
        self.nodeblack[1].white = self.nodewhite
        self.nodeblack[2].white = self.nodewhite
        self.nodeblack[3].white = self.nodewhite
        self.nodeblock = []
        self.turn = turn
        self.c = 1  # 参数
        self.rand = random.choices([9, 10], k=1)
        self.autoencoder1 = paras
        self.autoencoder2 = paras2
        self.time = minlimit

    def board(self):  # 返回数组对象
        return self.__board

    def boardchange(self, board):
        self.__board = board
        return

    # 对初始的四个棋子的节点进行可行步搜索
    def searchchildren(self, mode, temp=None):
        if mode == 0:
            for num in range(len(self.nodeblack)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j

                        while not (self.nodeblack[num].pos[0] + m < 0 or self.nodeblack[num].pos[
                            1] + n < 0 or
                                   self.nodeblack[num].pos[0] + m > 9 or self.nodeblack[num].pos[
                                       1] + n > 9):
                            if self.__board[self.nodeblack[num].pos[0] + m][self.nodeblack[num].pos[1] + n] == EMPTY:
                                temp.append([[self.nodeblack[num].pos[0], self.nodeblack[num].pos[1]],
                                             [self.nodeblack[num].pos[0] + m, self.nodeblack[num].pos[1] + n], mode,
                                             -1, self.nodeblack[num]])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        elif mode == 1:
            for num in range(len(self.nodewhite)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (self.nodewhite[num].pos[0] + m < 0 or self.nodewhite[num].pos[
                            1] + n < 0 or
                                   self.nodewhite[num].pos[0] + m > 9 or self.nodewhite[num].pos[
                                       1] + n > 9):
                            if self.__board[self.nodewhite[num].pos[0] + m][self.nodewhite[num].pos[1] + n] == EMPTY:
                                temp.append([[self.nodewhite[num].pos[0], self.nodewhite[num].pos[1]],
                                             [self.nodewhite[num].pos[0] + m, self.nodewhite[num].pos[1] + n], mode,
                                             -1, self.nodewhite[num]])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        return temp

    # 根据mode类型计算self.objmove函数的值
    def update1(self, onenode, turns, boardcopy):
        if onenode[3] != -1:
            return onenode[3]
        boardcopycopy = copy.deepcopy(boardcopy)
        if onenode[2] == 0:
            nodepos = copy.deepcopy(onenode[4].black)
            for count in range(len(onenode[4].black)):
                if onenode[4].black[count].pos == onenode[0]:
                    boardcopycopy[onenode[0][0]][onenode[0][1]] = EMPTY
                    boardcopycopy[onenode[1][0]][onenode[1][1]] = BLACK
                    nodepos[count].pos = onenode[1]
                    break
            if turns <= 30:
                a, b, c, d, e = self.objmove(onenode[1], onenode[2], turns, boardcopycopy, onenode[4].white, nodepos)
            elif turns > 30:
                b, a, c, d, e = self.objmove(onenode[1], onenode[2], turns, boardcopycopy, onenode[4].white, nodepos)

        elif onenode[2] == 1:
            nodepos = copy.deepcopy(onenode[4].white)
            for count in range(len(onenode[4].white)):
                if onenode[4].white[count].pos == onenode[0]:
                    boardcopycopy[onenode[0][0]][onenode[0][1]] = EMPTY
                    boardcopycopy[onenode[1][0]][onenode[1][1]] = WHITE
                    nodepos[count].pos = onenode[1]
                    break
            if turns <= 30:
                a, b, c, d, e = self.objmove(onenode[1], onenode[2], turns, boardcopycopy, nodepos, onenode[4].black)
            elif turns > 30:
                b, a, c, d, e = self.objmove(onenode[1], onenode[2], turns, boardcopycopy, nodepos, onenode[4].black)
        onenode[3] = self.autoencoder1.forward1(torch.tensor([a, b, c, d, e], dtype=torch.float)).float()
        return onenode[3], [a, b, c, d, e]

    def valappend(self, feasible, turns, eps, i):

        # 计算原始UCB值（使用你的原始逻辑）
        obj, list1 = self.update1(feasible, turns, feasible[4].board)
        obj = obj.item()
        base_mud = obj + self.c * math.sqrt(2 * math.log2(self.totaltime) / (feasible[4].time + 1))

        return base_mud, i, list1, obj

    def valappend2(self, feasible, turns, eps, i, all_candidates, obj, GRPO_MCTS):
        # GRPO群体评估 这里全是0
        try:
            # 提取当前节点特征
            node_features = GRPO_MCTS.extract_node_features(feasible)
            # 分析群体状态
            group_state = GRPO_MCTS.analyze_group_state(all_candidates)
            # 搜索上下文
            search_context = {
                'turns': turns,
                'eps': eps,
                'total_time': self.totaltime,
                'progress': eps / turns if turns > 0 else 0.0,
                'node_index': i
            }
            # GRPO相对价值评估
            relative_adjustment = GRPO_MCTS.evaluate_node_in_group(
                node_features, group_state, search_context
            )
            grpo_agent = GRPO_MCTS
            last_c = getattr(grpo_agent, "last_adjusted_c", None)

            if last_c is not None:
                adjusted_c = float(last_c)
            else:
                # 否则按原来的动态调整（仍然可调用 grpo 的 adjust_exploration_constant）
                adjusted_c = GRPO_MCTS.adjust_exploration_constant(
                    base_c=self.c,
                    group_diversity=group_state[6].item() if len(group_state) > 6 else 0.0,
                    consensus_level=1.0 - group_state[7].item() if len(group_state) > 7 else 0.0,
                    search_progress=search_context['progress']
                )
            # 重新计算UCB with adjusted C
            exploration_term = adjusted_c * math.sqrt(2 * math.log2(self.totaltime) / (feasible[4].time + 1))

            # 获取来自 TCN+PV 的 per-candidate bias（如果存在）
            bias = 0.0
            try:
                biases = getattr(grpo_agent, "next_round_biases", None)
                if biases is not None:
                    # biases 存的是 index->bias，i 是当前候选在 all_candidates 中的索引
                    bias = float(biases.get(i, 0.0)) if isinstance(biases, dict) else 0.0
            except Exception:
                bias = 0.0

            # 最终得分：原始 obj + exploration + 群体相对调整 + TCN/PV bias
            enhanced_mud = obj + exploration_term + relative_adjustment + bias
            print(enhanced_mud)
            return enhanced_mud

        except Exception as e:
            print(f"GRPO增强评估失败，回退到原始UCB: {e}")
            return obj

    def lottery(self, feasible, turns, eps=0.001, length=0, GRPO_MCTS=None):  # 找出可行步中的最大val节点
        if len(feasible) == 0:
            return -1, -1, -1, -1, -1
        pool = ProcessPoolExecutor(max_workers=cpu_count() - 8)

        # UCB
        num = -1
        value = 0
        # 1.计算val
        # 2.计算UCB值，并根据UCB值选出最佳点 feasible:[[原位置，现位置, mode, val, 上一个节点],...]
        count = 0
        list1 = list()
        start_time = time.time()

        # 求新节点的值:
        for i in range(length, len(feasible)):
            list1.append(pool.submit(self.valappend, feasible[i], turns, eps, i))

        # feasible1 和 val中
        for res in as_completed(list1):
            a, b, c, obj = res.result()
            feasible[b][3] = a
            feasible[b][4].weight1 = c
            feasible[b][4].obj1 = obj
        pool.shutdown(wait=True)
        val = []
        if GRPO_MCTS:
            for i in range(len(feasible)):
                example = feasible[i]
                val.append(self.valappend2(example, turns, eps, i, feasible, feasible[i][4].obj1, GRPO_MCTS))

            def group_normalize(values, eps=1e-8):
                values = np.array(values, dtype=np.float32)
                mean = values.mean()
                std = values.std()
                return (values - mean) / (std + eps)

            val1 = group_normalize(val)
            print(val1)

            print("主进程结束耗时%s" % (time.time() - start_time))
            # softmax
            num = random.choices(list(range(len(feasible))), softmax(val1))[0]
            val2 = val1[num]
            value = feasible[num][3]
        else:
            print(feasible)
            num = random.choices(list(range(len(feasible))), softmax([feasible[i][3] for i in range(len(feasible))]))[0]
            temp = [feasible[i][3] for i in range(len(feasible))]


            def group_normalize(values, eps=1e-8):
                values = np.array(values, dtype=np.float32)
                mean = values.mean()
                std = values.std()
                return (values - mean) / (std + eps)

            val2 = group_normalize(temp)[num]
            value = feasible[num][3]

        # 更新移动的位置
        temp = copy.deepcopy(feasible[num][4].board)
        temp[feasible[num][1][0]][feasible[num][1][1]] = WHITE if temp[feasible[num][0][0]][
                                                                      feasible[num][0][1]] == WHITE else BLACK
        temp[feasible[num][0][0]][feasible[num][0][1]] = 0
        print('-------')
        # 更新新的节点
        update(feasible[num][4], feasible[num][1], feasible[num][0], feasible[num][4].weight1, feasible[num][2], temp)
        feasible[num][4].children[-1].val = value
        feasible[num][4].children[-1].obj = val2
        feasible[num][4].children[-1].step = [feasible[num][0], feasible[num][1]]
        feasible123 = list()
        self.searchput(feasible[num][1], temp, feasible123)
        a, b, c = self.objput(feasible[num][2], turns, feasible123, feasible[num][4].children[-1].board,
                              feasible[num][4].children[-1].white, feasible[num][4].children[-1].black)
        # 更新新的Block
        feasible[num][4].children[-1].board[feasible123[b][0]][feasible123[b][1]] = Block
        feasible[num][4].children[-1].put = feasible123[b]
        feasible[num][4].children[-1].block.append([feasible123[b][0], feasible123[b][1]])
        feasible[num][4].children[-1].weightput = c
        feasible[num][4].children[-1].feasible = feasible[num]
        feasible[num][4].children[-1].feasible_nodes = feasible
        feasible[num][4].children[-1].index = num

        # todo 这里还可以调整
        feasible[num][4].children[-1].obj = feasible[num][4].obj1 * (1 - rate) + a * rate
        print(a)
        print(feasible[num][4].obj1)
        # 确定lottery选择的点后，更新parent节点
        key = 0
        feasible[num][4].timeplus()
        a, b = feasible[num][4].research()
        if self.totaltime >= self.time:
            print("已达指定搜索层数")
            key = 1
        self.totaltime = self.totaltime + 1
        print(self.totaltime)
        print("搜索数量：")
        print(len(feasible) - length)
        print("目前选择的节点目标函数值:")
        print(value)
        print(feasible[num][4].obj1)
        if key != 1:
            a, b = feasible[num][4].children[-1], 1 - feasible[num][2]
            del feasible[num]
            print(time.time())
            return a, b, len(feasible), val, feasible
        else:
            return -1, -1, -1, -1, -1

    # 用UCB算法搜索可行步，选择
    def nextnode_searching(self, turns, feasible, length=0, eps=0.001, GRPO_MCTS=None):
        # feasible: 所有可行解 [[],[]] , posnode=feasible[num]

        posnode, modetemp, length, val, feasible = self.lottery(feasible, turns, eps, length, GRPO_MCTS)
        print(time.time())
        if posnode == -1:
            return -1, -1
        else:
            return posnode.search(modetemp, feasible), length

    def draw_xy(self, x, y, mode, lastx=-1, lasty=-1):  # 获取落子点坐标的状态
        if mode == 2:
            self.nodeblock.append([x, y])
            self.__board[x][y] = Block
            return
        elif mode == 0:
            for i in range(len(self.nodeblack)):
                if self.nodeblack[i].pos == [lastx, lasty]:
                    self.nodeblack[i].pos = [x, y]
                    self.__board[lastx][lasty] = EMPTY
                    self.__board[x][y] = BLACK
                    return
        elif mode == 1:
            for i in range(len(self.nodewhite)):
                if self.nodewhite[i].pos == [lastx, lasty]:
                    self.nodewhite[i].pos = [x, y]
                    self.__board[lastx][lasty] = EMPTY
                    self.__board[x][y] = WHITE
                    return

    def get_xy_on_logic_state(self, i, j):  # 获取指定点坐标的状态
        if self.__board[i][j] == EMPTY:
            return True
        else:
            return False

    def judge_xy_on_logic_state(self, i, j, lasti, lastj):
        if i == lasti or j == lastj:
            return 1
        if abs(lasti - i) == abs(lastj - j):
            return 1
        else:
            return 0

    def reset(self):  # 重置
        self.__board = [[EMPTY for n in range(10)] for m in range(10)]

    def objmove(self, pos, mode, turns, boardcopy, whitetemp, blacktemp):  # pos传入位置
        if turns < 30:
            return self.oneterritory(mode, whitetemp, blacktemp, boardcopy), self.lineterritory(mode, whitetemp,
                                                                                                blacktemp,
                                                                                                boardcopy), self.broad(
                pos, boardcopy), self.divergence(whitetemp, blacktemp, mode, boardcopy), self.position(mode, turns)
        elif turns >= 30:
            return self.lineterritory(mode, whitetemp, blacktemp, boardcopy), self.oneterritory(mode, whitetemp,
                                                                                                blacktemp,
                                                                                                boardcopy), self.broad(
                pos, boardcopy), self.divergence(whitetemp, blacktemp, mode, boardcopy), self.position(mode, turns)

    def searchput(self, movepos, boardcopy, feasible):
        for i in range(-1, 2):
            for j in range(-1, 2):
                multiply = 1
                m = i
                n = j
                while not (movepos[0] + m < 0 or movepos[1] + n < 0 or movepos[0] + m > 9 or movepos[1] + n > 9 or
                           boardcopy[movepos[0] + m][movepos[1] + n] != EMPTY):
                    feasible.append([movepos[0] + m, movepos[1] + n])
                    multiply = multiply + 1
                    m = i * multiply
                    n = j * multiply

    def objput(self, mode, turns, feasible, boardcopy, whitetemp,
               blacktemp):  # whitetemp:白的位置 blacktemp:黑棋的位置[[],[],[],[]] feasible:[]->放置的位置
        temp = []
        temp1 = []
        for i in range(len(feasible)):
            board = copy.deepcopy(boardcopy)
            board[feasible[i][0]][feasible[i][1]] = Block
            if turns < 30:
                a = self.oneterritory(mode, whitetemp, blacktemp, board)
                b = self.lineterritory(mode, whitetemp, blacktemp, board)
                c = self.broad2(whitetemp, blacktemp, board, mode)
                d = self.divergence(whitetemp, blacktemp, mode, board)
                e = self.position(mode, turns)
                temp.append(self.autoencoder2.forward1(torch.tensor([a, b, c, d, e], dtype=torch.float)).item())
            elif turns >= 30:
                b = self.lineterritory(mode, whitetemp, blacktemp, board)
                a = self.oneterritory(mode, whitetemp, blacktemp, board)
                c = self.broad2(whitetemp, blacktemp, board, mode)
                d = self.divergence(whitetemp, blacktemp, mode, board)
                e = self.position(mode, turns)
                temp.append(self.autoencoder2.forward1(torch.tensor([a, b, c, d, e], dtype=torch.float)).item())
            temp1.append([a, b, c, d, e])
        num = random.choices(list(range(len(feasible))), softmax(temp))[0]
        return temp[num], num, temp1[num]

    # 以下都是评估指标

    # 移动指标
    # 一个个搜索 返回可到达的领地距离和以及黑白双方格子的占据数
    def oneterritory(self, mode, chesswhite, chessblack, boardcopy):  # territory pos=[[原位置],[现位置]]
        temp = []
        self.white = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        # chesswhite保存着四个白子的位置 换句话说这里需要四个白子的位置
        self.black = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        boardco = copy.deepcopy(boardcopy)
        for white in chesswhite:
            self.white[white.pos[0]][white.pos[1]] = 0
            temp.append([white.pos[0], white.pos[1], 0])
            while len(temp) != 0:
                num = temp[0][2]
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        if not (temp[0][0] + i < 0 or temp[0][1] + j < 0 or temp[0][0] + i > 9 or temp[0][1] + j > 9 or
                                boardco[temp[0][0] + i][temp[0][1] + j] != 0) and boardco[temp[0][0] + i][
                            temp[0][1] + j] != -2:
                            if num + 1 < self.white[temp[0][0] + i][temp[0][1] + j]:
                                self.white[temp[0][0] + i][temp[0][1] + j] = num + 1
                                temp.append([temp[0][0] + i, temp[0][1] + j, num + 1])
                                boardco[temp[0][0] + i][temp[0][1] + j] = -2
                del temp[0]
        temp.clear()
        boardco = copy.deepcopy(boardcopy)
        for black in chessblack:
            self.black[black.pos[0]][black.pos[1]] = 0
            temp.append([black.pos[0], black.pos[1], 0])
            while len(temp) != 0:
                num = temp[0][2]
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        if not (temp[0][0] + i < 0 or temp[0][1] + j < 0 or temp[0][0] + i > 9 or temp[0][1] + j > 9 or
                                boardco[temp[0][0] + i][temp[0][1] + j] != 0) and boardco[temp[0][0] + i][
                            temp[0][1] + j] != -2:
                            if num + 1 < self.black[temp[0][0] + i][temp[0][1] + j]:
                                self.black[temp[0][0] + i][temp[0][1] + j] = num + 1
                                temp.append([temp[0][0] + i, temp[0][1] + j, num + 1])
                                boardco[temp[0][0] + i][temp[0][1] + j] = -2
                del temp[0]
        black1 = 0
        white1 = 0
        for i in range(10):
            for j in range(10):
                if self.white[i][j] > self.black[i][j]:
                    black1 = black1 + 1
                elif self.white[i][j] < self.black[i][j]:
                    white1 = white1 + 1
                else:
                    black1 = black1 + 1
                    white1 = white1 + 1
        if mode == 0:
            return black1 / white1
        elif mode == 1:
            return white1 / black1

    # 一连串搜索 返回可到达的领地距离和以及黑白双方格子的占据数
    def lineterritory(self, mode, chesswhite, chessblack, boardcopy):  # 全局territory
        temp = []
        self.white = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        self.black = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        boardco = copy.deepcopy(boardcopy)
        for white in chesswhite:
            self.white[white.pos[0]][white.pos[1]] = 0
            temp.append([white.pos[0], white.pos[1], 0])
            while len(temp) != 0:
                num = temp[0][2]
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (temp[0][0] + m < 0 or temp[0][1] + n < 0 or temp[0][
                            0] + m > 9 or temp[0][1] + n > 9 or
                                   self.white[temp[0][0] + m][temp[0][1] + n] == -1) and boardco[temp[0][0] + m][
                            temp[0][1] + n] != -2:
                            if num + 1 < self.white[temp[0][0] + m][temp[0][1] + n]:
                                self.white[temp[0][0] + m][temp[0][1] + n] = num + 1
                                temp.append([temp[0][0] + m, temp[0][1] + n, num + 1])
                                boardco[temp[0][0] + m][temp[0][1] + n] = -2
                            multiply = multiply + 1
                            m = i * multiply
                            n = j * multiply
                del temp[0]
        temp.clear()
        boardco = copy.deepcopy(boardcopy)
        for black in chessblack:
            self.black[black.pos[0]][black.pos[1]] = 0
            temp.append([black.pos[0], black.pos[1], 0])
            while len(temp) != 0:
                num = temp[0][2]
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (temp[0][0] + m < 0 or temp[0][1] + n < 0 or temp[0][
                            0] + m > 9 or temp[0][1] + n > 9 or
                                   self.black[temp[0][0] + m][temp[0][1] + n] == -1) and boardco[temp[0][0] + m][
                            temp[0][1] + n] != -2:
                            if num + 1 < self.black[temp[0][0] + m][temp[0][1] + n]:
                                self.black[temp[0][0] + m][temp[0][1] + n] = num + 1
                                temp.append([temp[0][0] + m, temp[0][1] + n, num + 1])
                                boardco[temp[0][0] + m][temp[0][1] + n] = -2
                            multiply = multiply + 1
                            m = i * multiply
                            n = j * multiply
                del temp[0]
        temp.clear()
        black2 = 0
        white2 = 0
        for i in range(10):
            for j in range(10):
                if self.white[i][j] > self.black[i][j]:
                    black2 = black2 + 1
                elif self.white[i][j] < self.black[i][j]:
                    white2 = white2 + 1
                else:
                    black2 = black2 + 1
                    white2 = white2 + 1
        if mode == 0:
            return black2 / white2
        elif mode == 1:
            return white2 / black2

    # 周围可行走数量 返回周围可行走的数量
    # 传入一个下一步下的棋子的位置
    def broad2(self, chesswhite, chessblack, boardcopy, mode):
        count = 0
        if mode == 0:
            chess = chesswhite
        elif mode == 1:
            chess = chessblack
        for chesstemp in chess:
            count += self.broad(chesstemp.pos, boardcopy)
        return count / 4

    def broad(self, chesspos, boardcopy):  # [2,2] #局部territory chesspos=[后位置]
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if not (chesspos[0] + i < 0 or chesspos[1] + j < 0 or chesspos[0] + i > 9 or chesspos[1] + j > 9):
                    if boardcopy[chesspos[0] + i][chesspos[1] + j] == EMPTY:
                        count = count + 1
        return count / 8

    # 传统position度量
    def position(self, mode, turns):  # 全局territory   白在前 黑在后
        sum1 = 0
        if mode == 0:
            for i in range(10):
                for j in range(10):
                    if turns < 30:
                        sum1 = sum1 + math.pow(2, self.black[i][j]) - math.pow(2, self.white[i][j])
                    if turns >= 30:
                        sum1 = sum1 + min(1, max(-1, (self.black[i][j] - self.white[i][j])))
        elif mode == 1:
            for i in range(10):
                for j in range(10):
                    if turns < 30:
                        sum1 = sum1 + math.pow(2, self.white[i][j]) - math.pow(2, self.black[i][j])
                    if turns >= 30:
                        sum1 = sum1 + min(1, max(-1, (self.white[i][j] - self.black[i][j])))
        if sum1 > 100000 or sum1 < -100000:
            return 0
        else:
            return sum1 / 100

    # 计算他的信息增益(right) 返回信息熵 最大为2
    # 传入4+4个棋子的位置
    def divergence(self, chesswhite, chessblack, mode, boardcopy):  # 聚集程度  确保可到位置的均衡性
        temp = []
        white2 = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        black2 = [[999 if boardcopy[m][n] == 0 else -1 for n in range(10)] for m in range(10)]
        for white in chesswhite:
            white2[white.pos[0]][white.pos[1]] = 0
            temp.append([white.pos[0], white.pos[1]])
            while len(temp) != 0:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (
                                temp[0][0] + m < 0 or temp[0][1] + n < 0 or temp[0][0] + m > 9 or temp[0][1] + n > 9 or
                                white2[temp[0][0] + m][temp[0][1] + n] == -1):
                            if white2[temp[0][0] + m][temp[0][1] + n] > white2[temp[0][0]][temp[0][1]] + 1:
                                white2[temp[0][0] + m][temp[0][1] + n] = white2[temp[0][0]][temp[0][1]] + 1
                                temp.append([temp[0][0] + m, temp[0][1] + n])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
                del temp[0]
        temp.clear()
        for black in chessblack:
            black2[black.pos[0]][black.pos[1]] = 0
            temp.append([black.pos[0], black.pos[1]])
            while len(temp) != 0:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (
                                temp[0][0] + m < 0 or temp[0][1] + n < 0 or temp[0][0] + m > 9 or temp[0][1] + n > 9 or
                                black2[temp[0][0] + m][temp[0][1] + n] == -1):
                            if black2[temp[0][0] + m][temp[0][1] + n] > black2[temp[0][0]][temp[0][1]] + 1:
                                black2[temp[0][0] + m][temp[0][1] + n] = black2[temp[0][0]][temp[0][1]] + 1
                                temp.append([temp[0][0] + m, temp[0][1] + n])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
                del temp[0]
        temp.clear()
        whitesum = 0
        blacksum = 0
        # 信息增益计算
        white1 = np.array(white2).flatten()
        black1 = np.array(black2).flatten()
        whiteone = sum(white1[i] == 1 for i in range(100)) / 100
        whitetwo = sum(white1[i] == 2 for i in range(100)) / 100
        whitethree = sum(white1[i] == 3 for i in range(100)) / 100
        whitefour = sum(white1[i] == 4 for i in range(100)) / 100
        blackone = sum(black1[i] == 1 for i in range(100)) / 100
        blacktwo = sum(black1[i] == 2 for i in range(100)) / 100
        blackthree = sum(black1[i] == 3 for i in range(100)) / 100
        blackfour = sum(black1[i] == 4 for i in range(100)) / 100
        for i in range(10):
            for j in range(10):
                if white2[i][j] == 1:
                    if whiteone != 0:
                        whitesum = whitesum + whiteone * (-1) * math.log2(whiteone)
                elif white2[i][j] == 2:
                    if whitetwo != 0:
                        whitesum = whitesum + whitetwo * (-1) * math.log2(whitetwo)
                elif white2[i][j] == 3:
                    if whitethree != 0:
                        whitesum = whitesum + whitethree * (-1) * math.log2(whitethree)
                elif white2[i][j] == 4:
                    if whitefour != 0:
                        whitesum = whitesum + whitefour * (-1) * math.log2(whitefour)
                if black2[i][j] == 1:
                    if blackone != 0:
                        blacksum = blacksum + blackone * (-1) * math.log2(blackone)
                elif black2[i][j] == 2:
                    if blacktwo != 0:
                        blacksum = blacksum + blacktwo * (-1) * math.log2(blacktwo)
                elif black2[i][j] == 3:
                    if blackthree != 0:
                        blacksum = blacksum + blackthree * (-1) * math.log2(blackthree)
                elif black2[i][j] == 4:
                    if blackfour != 0:
                        blacksum = blacksum + blackfour * (-1) * math.log2(blackfour)
        if mode == 1:
            return whitesum / 50
        elif mode == 0:
            return blacksum / 50

    # 放置指标
    # 连续性的度量 看看某个点放上障碍时，障碍的连续度会占据棋盘的多少 返回比例【0,1】
    # 传入障碍放置的坐标
    def continuous(self, blockpos, boardcopy):  # 重大位置 (good)
        board = copy.deepcopy(boardcopy)
        x = blockpos[0]
        y = blockpos[1]
        board[x][y] = Block
        temp = [[x, y, 0, 0, 0, 0]]
        left = 0
        right = 0
        top = 0
        bottom = 0
        while len(temp) != 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if not (temp[0][0] + i < 0 or temp[0][1] + j < 0 or temp[0][0] + i > 9 or temp[0][1] + j > 9 or
                            board[temp[0][0] + i][temp[0][1] + j] != Block):
                        list1 = [temp[0][0] + i, temp[0][1] + j, temp[0][2], temp[0][3], temp[0][4], temp[0][5]]
                        if i == 1:
                            list1[3] = list1[3] + 1
                        elif i == -1:
                            list1[2] = list1[2] + 1
                        if i == 1:
                            list1[4] = list1[4] + 1
                        elif i == -1:
                            list1[5] = list1[5] + 1
                        if list1[2] > left:
                            left = list1[2]
                        if list1[3] > right:
                            right = list1[3]
                        if list1[4] > top:
                            top = list1[4]
                        if list1[5] > bottom:
                            bottom = list1[5]
                        temp.append(list1.copy())
                        board[temp[0][0] + i][temp[0][1] + j] = 4
            del temp[0]
        del board
        return (right + left) * (top - bottom) / 10

    # 障碍物可部署到的数量的比例 返回比例 [0,1]
    # mode=0时，计算白棋；mode=1时，计算黑棋
    def onestepblock(self, movepos, boardcopy):  # 障碍物可部署到的数量的比例
        feasible = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                multiply = 1
                m = i
                n = j
                while not (movepos[0] + m < 0 or movepos[1] + n < 0 or movepos[0] + m > 9 or movepos[1] + n > 9 or
                           boardcopy[movepos[0] + m][movepos[1] + n] != 0):
                    feasible.append([movepos[0] + i, movepos[1] + j])
                    multiply = multiply + 1
                    m = i * multiply
                    n = j * multiply
        num = len(feasible)
        feasible.clear()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                multiply = 1
                m = i
                n = j
                while not (movepos[0] + m < 0 or movepos[1] + n < 0 or movepos[0] + m > 9 or movepos[1] + n > 9):
                    feasible.append([movepos[0] + m, movepos[1] + n])
                    multiply = multiply + 1
                    m = i * multiply
                    n = j * multiply
        numa = len(feasible)
        return num / numa

    # 遗传算法开启
    def geneticinit(self, threshold, temp, mode, origin):  # temp中有两个可行节点
        a = [i.obj for i in temp]
        minimum = min([i.obj for i in temp])
        if minimum < 0:
            a = softmax(list(map(lambda x: x - minimum, a)))
        new = random.choices(temp, a, k=2)
        # 变异: 回退步 biased random walking
        num1 = new[0].genemutation(temp, origin)
        num2 = new[1].genemutation(temp, origin)
        # 交叉 添加了一个子节点
        new[0].genecross(new[1], temp)
        # 假如选择的节点无子节点
        # 终止条件
        if new[0].current != 0:
            print(new[0].current)
            if new[0].current > 50000:
                return None, 2
            print(new[0].height, Node.heightmax - new[0].height + 1)
            if new[0].current >= math.pow(2, Node.heightmax - new[0].height + 1) and new[0].height >= 1:
                temp = new[0]
                if len(temp.children) != 0:
                    while temp.children:
                        a = [i.obj for i in temp.children]
                        minimum = min([i.obj for i in temp.children])
                        if minimum < 0:
                            a = softmax(list(map(lambda x: x - minimum, a)))
                        temp = random.choices(temp.children, a)[0]
                    temp.current += 1
                    return new[0], 1
                else:
                    return new[0], 0
            else:
                return random.choices([new[0], new[1]])[0], 1
        # 否则
        elif new[1].current != 0:
            print(new[1].current)
            if new[1].current > 50000:
                return None, 2
            print(new[1].height, Node.heightmax - new[1].height + 1)
            if new[1].current >= math.pow(2, Node.heightmax - new[1].height + 1) and new[1].height >= 1:
                temp = new[1]
                if len(temp.children) != 0:
                    while temp.children:
                        a = [i.obj for i in temp.children]
                        minimum = min([i.obj for i in temp.children])
                        if minimum < 0:
                            a = softmax(list(map(lambda x: x - minimum, a)))
                        temp = random.choices(temp.children, a)[0]
                    temp.current += 1
                    return new[1], 1
                else:
                    return new[1], 0
            else:
                return random.choices([new[0], new[1]])[0], 1
