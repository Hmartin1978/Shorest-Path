import numpy as np
import random

# # 定义有向加权图的邻接矩阵表示


graph = {
    'A': {'B': 5, 'C': 1},
    'B': {'C': 2, 'D': 1},
    'C': {'D': 4, 'E': 8},
    'D': {'E': 3, 'F': 6},
    'E': {},
    'F': {}
}

#获取一个格子的状态
def get_state(state:str):
    return 

#在一个格子里做一个动作
def move(current_state:int, action:int, target_state:int):
    #如果当前已经在陷阱或者终点，则不能执行任何动作
    if current_state == target_state:
        return current_state, 0

    new_state = action

    reward = Q[current_state][action]
    if Q[current_state][action] > 1000000:
        reward = 100

    return new_state, reward

#根据状态选择一个动作
def get_action(state:int):
    #有小概率选择随机动作
    if random.random() < 0.1:
        return random.choice(range(4))

    #否则选择权重最低的路径
    return min(enumerate(Q[state]), key=lambda x : x[1])[0]

#更新分数，每次更新取决于当前的格子，当前的动作，下个格子，和下个格子的动作
def get_update(current_state, action, reward, new_state, next_action): # st, at, rt+1, st+1, at+1
    # 对照笔记"A unified point of view"来看

    #计算target
    target = 0.9 * Q[new_state, next_action]
    target += reward

    #计算value
    value = Q[current_state, action]

    #根据时序差分算法,当前state,action的分数 = 下一个state,action的分数*gamma + reward
    #此处是求两者的差,越接近0越好
    update = target - value

    #这个0.1相当于lr,即αt
    update *= 0.1

    #更新当前状态和动作的分数
    return update

#初始化在每一个格子里采取每个动作的分数,初始化都是0,因为没有任何的知识
# Q = np.zeros([4, 12, 4])


#训练
def train(start_state, target_state):
    for epoch in range(1500):
        #初始化当前位置
        # row = random.choice(range(4))
        # col = 0
        current_state = start_state

        #初始化第一个动作
        action = get_action(current_state)

        #计算反馈的和，这个数字应该越来越小
        reward_sum = 0

        #循环直到到达终点或者掉进陷阱
        while get_state(current_state) != target_state:

            #执行动作
            new_state, reward = move(current_state, action, target_state) # s(t+1), r(t+1) = move(st, at)
            reward_sum += reward

            #求新位置的动作
            next_action = get_action(new_state) # a(t+1) = get_action(s(t+1))

            #更新分数
            update = get_update(current_state, action, reward, new_state,
                                next_action)
            Q[current_state, action] += update

            #更新当前位置
            current_state = new_state
            action = next_action

        if epoch % 150 == 0:
            print(epoch, reward_sum)

start = 0
target = 4
Q = np.array([
    [np.inf, 5, 1, np.inf, np.inf, np.inf],
    [np.inf, np.inf, 2, 1, np.inf, np.inf],
    [np.inf, np.inf, np.inf, 4, 8, np.inf],
    [np.inf, np.inf, np.inf, np.inf, 3, 6],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
])

train(start, target)
