import math
import random
import copy
import torch
import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2
Block = 3
import sys
import time
from collections import defaultdict

sys.setrecursionlimit(3000)  # 修改递归深度为 3000
random.seed(time.time())

from openai import OpenAI

api_key = 'sk-IxoidByfpUpfmhVbMiDdH98oclTSufnOIT4lLGyjEPjoyLzK'
client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")


def softmax(list1):
    return np.exp(np.array(list1)) / sum(np.exp(np.array(list1)))


def count_children(node):
    if len(node.children) == 0:
        return 1
    total_children = 0
    for child in node.children:
        total_children += count_children(child)
    return total_children


class Node(object):
    clock = -1
    heightmax = -1
    count = 1

    def __init__(self, parent, pos, boardcopy, whitenode, blacknode, block, val: int = 0, max: bool = True, time1=0,
                 mode=None, oneterritory=0, lineterritory=0, broad=0, divergence=0, position=0, height=0
                 ) -> None:
        '''
        val: 节点值
        max: 是否为max层
        childern: 子节点列表
        '''
        self.height = height
        self.val = val  # 该节点的值（目标函数） 不变的
        self.obj = 0.0  # objmove的值 预计还要加上objput的值
        self.pos = pos  # 该节点的代表位置 不变
        self.max = max  # 该层是否为max层，默认顶层节点为max层 不变的
        self.children: list = []  # 该节点的子节点 不断扩展
        self.parent = parent  # 父节点
        self.time = time1  # 多变
        self.board = boardcopy  # 记得更新
        self.white = whitenode  # 白子节点
        self.black = blacknode  # 黑子节点
        self.mode = mode
        self.block = block  # 所有block的列表
        self.current = 0
        self.obj1 = 0
        self.step = []
        self.put = []
        self.weight = [oneterritory, lineterritory, broad, divergence, position]
        self.weight1 = []
        self.weightput = []
        self.number = Node.count
        self.feasible = None
        self.feasible_nodes=None
        self.index = None

        # —— MCTS/RL 相关 ——
        self.state = None  # 当前 MDP 状态
        self.N = defaultdict(int)  # 访问计数 N(s,a)
        self.W = defaultdict(float)  # 累计价值 W(s,a)
        self.P = {}  # 先验概率 P(s,a)，在 Expansion 时填入
        self.mcts_children = {}  # parallel list: 存对应的 action_idx

    def select(self, c_puct=1.0):
        """UCB选择策略"""
        if not self.P:
            return 0  # 默认返回第一个动作

        return max(self.P.keys(), key=lambda a:
        (self.W[a] / (1 + self.N[a])) +
        c_puct * self.P[a] * math.sqrt(sum(self.N.values())) / (1 + self.N[a])
                   )

    def expand(self, action_probs):
        """展开节点，设置先验概率"""
        for i, prob in enumerate(action_probs):
            self.P[i] = prob

    def backup(self, action, reward):
        """回传奖励值"""
        self.N[action] += 1
        self.W[action] += reward

    def add_child(self, pos, val, max, boardcopy, whitenode, blacknode, paras, time1=0, mode=None):  # pos
        self.children.append(
            Node(copy.copy(self), pos, boardcopy, whitenode, blacknode, copy.deepcopy(self.block), val, 1 - self.max,
                 time1,
                 mode, paras[0], paras[1], paras[2], paras[3], paras[4], self.height + 1))
        Node.count += 1

    def add_mcts_child(self, action_idx, child_node):
        """把实际生成的 child_node 关联到 action_idx 上。"""
        self.mcts_children[action_idx] = child_node

    def timeplus(self):
        self.time += 1
        if self.height > Node.heightmax:
            Node.heightmax = self.height
        if isinstance(self.parent, list):
            return
        else:
            if self.parent:
                return self.parent.timeplus()
            else:
                return

    def postorder_traversal(self, array, turn, fromid, myid, idbase, adjacency, arrayencoder1, arrayencoder2, newlist):
        for child in self.children:
            temp1 = idbase.pop(0)
            child.postorder_traversal(array, turn + 1, fromid + [myid], temp1, idbase, adjacency, arrayencoder1,
                                      arrayencoder2, newlist)
        for i in range(1, len(fromid) + 1):
            adjacency[0][fromid[i - 1] - 1][myid - 1] = len(fromid) + 1 - i
        if myid != 1:
            if self.mode == 0:
                chess = "黑"
            else:
                chess = "白"
            content = "下面是一个亚马逊棋的棋盘:\n" + str(
                self.board) + "\n其中，1代表白棋，2代表黑棋，3代表一个阻挡。" + chess + "棋将要从" + str(
                self.step[0]) + "移动到" + str(self.step[1]) + ",并在" + str(self.put) + ("放置一个障碍。\n"
                                                                                          "这是棋盘的信息，分数越高说明这一步更有利于取得胜利。Instruction: 首先考虑一下亚马逊棋的规则。\n请分别给出这一步") + chess + "棋移动对应的[0,1](0至1的区间)的评分，用于评估" + chess + "棋的这步的好坏；以及这一步障碍对应的[0,1](0至1的区间)的评分，用于评估障碍的好坏。只输出评分，格式为：\n评分1 评分2\n但这并不代表移动一步和放置障碍的评分总和为1，这两个行为的评分是独立的。不要输出多余的信息："
            temp = torch.tensor([self.weight], dtype=torch.float32)
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
                    max_tokens=1024,
                    top_p=1,
                )
                try:
                    val = response.choices[0].message.content.strip().split(" ")
                    val1 = float(val[0])
                    val2 = float(val[1])
                    break
                except:
                    print("mistake")
                    print(str(response.choices[0].message.content))
                    continue
            val = self.mode

            # 这个要注释
            arrayencoder1[0][myid - 1][:] = torch.tensor(self.weight + [val1])
            arrayencoder2[0][myid - 1][:] = torch.tensor(self.weightput + [val2])

            temp = torch.cat(
                (torch.sum(temp, dim=0) / temp.size()[0], torch.tensor([turn, val], dtype=torch.float32)),
                dim=0)  # turn表示第几个回合了 self.max为y值
            array[0][myid - 1, :] = temp
            newlist[myid - 1] = self

    def research(self):
        if not self.parent:
            print("搜索次数为:")
            print(self.time)
            num = count_children(self)
            print("有子节点数:")
            print(num)
            return self.time / (num), num
        else:
            return self.parent.research()

    def search(self, mode, temp=None):
        if mode == 0:
            for num in range(len(self.black)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (self.black[num].pos[0] + m < 0 or self.black[num].pos[
                            1] + n < 0 or
                                   self.black[num].pos[0] + m > 9 or self.black[num].pos[
                                       1] + n > 9):
                            if self.board[self.black[num].pos[0] + m][self.black[num].pos[1] + n] == EMPTY:
                                temp.append([[self.black[num].pos[0], self.black[num].pos[1]],
                                             [self.black[num].pos[0] + m, self.black[num].pos[1] + n], mode, -1, self])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        elif mode == 1:
            for num in range(len(self.white)):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        multiply = 1
                        m = i
                        n = j
                        while not (self.white[num].pos[0] + m < 0 or self.white[num].pos[
                            1] + n < 0 or
                                   self.white[num].pos[0] + m > 9 or self.white[num].pos[
                                       1] + n > 9):
                            if self.board[self.white[num].pos[0] + m][self.white[num].pos[1] + n] == EMPTY:
                                temp.append([[self.white[num].pos[0], self.white[num].pos[1]],
                                             [self.white[num].pos[0] + m, self.white[num].pos[1] + n], mode, -1, self])
                                multiply = multiply + 1
                                m = i * multiply
                                n = j * multiply
                            else:
                                break
        return temp

    def entrophy(self):  # 交叉熵  这里会进行minmax计算
        sum = 0
        for i in self.children:
            sum = sum + i.obj
        num = 0
        for i in self.children:
            num = num + i.obj / sum * math.log(i.obj / sum)
        return (-1) * num

    # 变异
    def genemutation(self, temp, origin, threshold=0.8):  # 这里可以设置参数优化
        self.current += 1
        if len(self.children) == 0:
            temp.clear()
            temp.extend(origin)
            print(self.current)
            print("no children!")
            return 1
        num = random.choices([0, 1], [1 - threshold, threshold])[0]
        if num == 1:
            a = [child.obj for child in self.children]
            minimum = min([child.obj for child in self.children])
            if minimum < 0:
                a = softmax(list(map(lambda x: x - minimum, a)))
            temp.append(random.choices(self.children, a)[0])
            print("children")
        elif num == 0:
            if self.parent is None:
                a = [i.obj for i in origin]
                minimum = min([i.obj for i in origin])
                if minimum < 0:
                    a = softmax(list(map(lambda x: x - minimum, a)))
                node = random.choices(origin, a)[0]
                a = [i.obj for i in node.children]
                minimum = min([i.obj for i in node.children])
                if minimum < 0:
                    a = softmax(list(map(lambda x: x - minimum, a)))
                temp.append(random.choices(node.children, a)[0])
                print("no parent")
                return
            temp.append(self.parent)
            print("parent")
        return 0

        # 交叉

    def genecross(self, anothertemp, temp):
        if self.pos == anothertemp.pos:
            try:
                temp.append(random.choices([self, anothertemp], [self.obj, anothertemp.obj])[0])
            except:
                a = [self.obj, anothertemp.obj]
                minimum = min([self.obj, anothertemp.obj])
                if minimum < 0:
                    a = softmax(list(map(lambda x: x - minimum, a)))
                temp.append(random.choices([self, anothertemp], a)[0])
        return

    def update_node_values(self, node, count):
        # 先处理子节点 todo
        for child in node.children:
            child.update_node_values(child, count + 1)
        # 计算子节点的value的平均值
        if node.children:  # 如果有子节点
            children_values = [child.obj for child in node.children]
            average_children_value = sum(children_values) / len(children_values)
            # 更新当前节点的value
            if count % 2 == 1:
                node.obj += 1 / pow(2, average_children_value)
            else:
                node.obj += average_children_value

    def update_node_values2(self, node):
        # 先处理子节点 todo
        for child in node.children:
            node.update_node_values2(child)
        # 计算子节点的value的平均值
        self.obj = self.obj / (self.heightmax + 1 - self.height)
        print(self.obj)

    def minmax(self, side):
        if len(self.children) == 0:
            return self.obj  # 使用评估函数返回得分
        if side == 0:  # 对手下棋
            a = 999999
            for i in self.children:
                a = min(a, i.minmax(1 - side))
        else:  # 我方下棋
            a = -999999
            for i in self.children:
                a = max(a, i.minmax(1 - side))
        return a

    def alphabeta(self, side, a, b):
        if len(self.children) == 0:
            return self.obj  # 使用评估函数返回得分
        if side == 1:  # beta剪枝
            for i in self.children:
                a = max(a, i.alphabeta(1 - side, a, b))
                if b <= a:
                    break
            return a
        else:  # alpha剪枝
            for i in self.children:
                b = min(b, i.alphabeta(1 - side, a, b))
                if b <= a:
                    break
            return b


class Tree(object):
    paras = [0, 0, 0, 0, 0]  # 初始化参数

    def __init__(self) -> None:
        self.root = Node(None, 0, [], 0, False)  # Node()，根节点 todo

    def build_tree(self, pos, root) -> None:  # 表示类型什么都不返回
        self.root.val = -1
