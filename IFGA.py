import copy
import math
import random
import time

import numpy as np


# 要根据调度方案变化的属性只有TC、Machine_list、ni、Ci、Cj和S
class IFGA_class:
    '''
    Attibute:
        NP:种群大小
        Pc:交叉概率
        Pm:变异概率
        m:机器数
        n:工件数
        processing_time:各工件在各机器上的加工时间
        release_time:各工件在机器上的释放时间
        weight:各工件在机器上的权重
        li:排序后的成本
        lamb:阈值控制参数
        r:惩罚参数
        Iter:迭代次数
        K:未提升最优解的次数上限
    '''

    def __init__(self, NP, Pc, Pm, m, n, processing_time, release_time, weight, li,
                 lamb, r, Iter, K):  # 参数确定后相当于一个实例
        self.NP = NP
        self.Pc = Pc
        self.Pm = Pm
        self.m = m
        self.n = n
        self.processing_time = processing_time
        self.release_time = release_time
        self.weight = weight
        self.li = li
        self.lamb = lamb
        self.r = r
        self.Iter = Iter
        self.K = K

    # √解码得Machine_list
    def decode(self, S):
        """
           对染色体S进行解码
           返回机器集Machine_list
           机器、工件索引从0开始，“值”从1开始
        """
        m = self.m
        n = self.n
        # 初始化机器集，每个机器对应一个空列表
        Machine_list = [[] for _ in range(m)]
        # 对染色体S进行排序，同时获取其索引
        sorted_indices = sorted(enumerate(S), key=lambda x: x[1])
        for index, value in sorted_indices:
            # 计算工件应该分配到的机器索引
            thisMachine = math.floor(value) - 1  # 保留整数部分
            # 将工件的索引加1后添加到相应机器的列表中
            Machine_list[thisMachine].append(index + 1)
        return Machine_list

    # √编码得S
    def encode(self, Machine_list):
        """
           对工件序列Machine_list进行编码
           返回染色体S
           机器、工件索引从0开始，“值”从1开始
        """
        m = self.m
        n = self.n
        ni = []
        for i in range(m):
            ni.append(len(Machine_list[i]))
        # 初始化染色体S
        S = [0] * n
        for i in range(m):
            if len(Machine_list[i]) > 0:
                # 生成len(Machine_list[i])个数量的不同的随机小数
                non_zero_decimal_j = [i * 0.001 for i in range(1, len(Machine_list[i]) + 1)]
                count = 0
                for element in Machine_list[i]:
                    S[element - 1] = non_zero_decimal_j[count] + i + 1
                    count += 1
            S = [round(num, 3) for num in S]
        return S

    # √计算单机器的完工时间。机器值thismachine从1开始（用于VNS搜索）
    def OneMachine_CompletionTime(self, Machine_list, thismachine):
        '''
        计算thismachine的完工时间及其工件序列的完工时间
        '''
        thismachine -= 1  # 转换成索引
        processing_time = self.processing_time
        release_time = self.release_time
        if len(Machine_list[thismachine]) == 0:
            Ci_thismachine = 0.0
        elif len(Machine_list[thismachine]) == 1:
            Ci_thismachine = release_time[Machine_list[thismachine][0] - 1] + processing_time[thismachine][
                Machine_list[thismachine][0] - 1]
        else:
            S = [0.0 for i in range(len(Machine_list[thismachine]))]  # 记录thismachine上第几个位置处工件在机器上的开始时间
            S[0] = release_time[Machine_list[thismachine][0] - 1]
            Ci_thismachine = S[0] + processing_time[thismachine][
                Machine_list[thismachine][0] - 1]
            for j in range(1, len(Machine_list[thismachine])):
                S[j] = max(S[j - 1] + processing_time[thismachine][Machine_list[thismachine][j - 1] - 1],
                           release_time[Machine_list[thismachine][j] - 1])
                Temp = S[j] + processing_time[thismachine][
                    Machine_list[thismachine][j] - 1]
                if Ci_thismachine < Temp:
                    Ci_thismachine = Temp
        return Ci_thismachine

    # √计算所有机器的完工时间Ci[]。
    def AllMachine_CompletionTime(self, Machine_list):
        Ci = []
        for i in range(self.m):
            Ci.append(self.OneMachine_CompletionTime(Machine_list, i + 1))
        return Ci

    # √计算所有工件的完工时间Cj[]。
    def AllJob_CompletionTime(self, Machine_list):
        processing_time = self.processing_time
        release_time = self.release_time
        Cj = [0.0 for i in range(self.n)]
        for thismachine in range(self.m):
            if len(Machine_list[thismachine]) == 1:
                Cj[Machine_list[thismachine][0] - 1] = release_time[Machine_list[thismachine][0] - 1] + \
                                                       processing_time[thismachine][Machine_list[thismachine][0] - 1]
            if len(Machine_list[thismachine]) > 1:
                S = [0.0 for i in range(len(Machine_list[thismachine]))]  # 记录thismachine上第几个位置处工件在机器上的开始时间
                S[0] = release_time[Machine_list[thismachine][0] - 1]
                Cj[Machine_list[thismachine][0] - 1] = S[0] + processing_time[thismachine][
                    Machine_list[thismachine][0] - 1]
                for j in range(1, len(Machine_list[thismachine])):
                    S[j] = max(S[j - 1] + processing_time[thismachine][Machine_list[thismachine][j - 1] - 1],
                               release_time[Machine_list[thismachine][j] - 1])
                    Cj[Machine_list[thismachine][j] - 1] = S[j] + processing_time[thismachine][
                        Machine_list[thismachine][j] - 1]
        return Cj

    # √计算单机器产生的WC，输入单机器值!!!thismachine以及该机器上的工件序列one_machine_list，输出one_WC
    def calculate_one_machine_WC(self, thismachine, one_machine_list):
        indexmachine = thismachine - 1  # 转化成索引
        processing_time = self.processing_time
        release_time = self.release_time
        weight = self.weight
        one_WC = 0.0
        if len(one_machine_list) == 0:
            return 0.0
        elif len(one_machine_list) == 1:
            Cj = [0.0 for i in range(len(one_machine_list))]
            Cj[0] = release_time[one_machine_list[0] - 1] + processing_time[indexmachine][one_machine_list[0] - 1]
            return Cj[0] * weight[one_machine_list[0] - 1]
        elif len(one_machine_list) > 1:
            Cj = [0.0 for i in range(len(one_machine_list))]
            S = [0.0 for i in range(len(one_machine_list))]  # 记录thismachine上第几个位置处工件在机器上的开始时间
            S[0] = release_time[one_machine_list[0] - 1]
            Cj[0] = S[0] + processing_time[indexmachine][one_machine_list[0] - 1]
            for j in range(1, len(one_machine_list)):
                S[j] = max(S[j - 1] + processing_time[indexmachine][one_machine_list[j - 1] - 1],
                           release_time[one_machine_list[j] - 1])
                Cj[j] = S[j] + processing_time[indexmachine][one_machine_list[j] - 1]
            for j in range(len(Cj)):
                one_WC += Cj[j] * weight[one_machine_list[j] - 1]
            return one_WC

    # √计算调度产生的总WC
    def calculate_WC(self, Machine_list):
        WC = 0.0
        Cj = self.AllJob_CompletionTime(Machine_list)
        weight = self.weight
        for j in range(len(Cj)):
            WC += Cj[j] * weight[j]
        return WC

    # √计算调度产生的资源消耗TC
    def calculate_TC(self, Machine_list):
        m = self.m
        processing_time = self.processing_time
        li = self.li
        TC = 0.0
        for i in range(m):
            for j in Machine_list[i]:
                TC += processing_time[i][j - 1] * li[i]
        return TC

    # √计算总资源约束U_total
    def calculate_U_total(self):
        processing_time = self.processing_time
        li = self.li
        lamb = self.lamb
        # 总资源阈值设置
        U_low = sum(np.min(np.array(li).reshape(-1, 1) * np.array(processing_time), axis=0))
        U_up = sum(np.max(np.array(li).reshape(-1, 1) * np.array(processing_time), axis=0))
        U_total = U_low + lamb * (U_up - U_low)
        return U_total

    # √计算对应的适应度值fitness
    def calculate_fitness(self, Machine_list):
        r = self.r
        U_total = self.calculate_U_total()
        TC = self.calculate_TC(Machine_list)
        WC = self.calculate_WC(Machine_list)
        penalty = r * (min(0, U_total - TC)) ** 2
        fitness = WC + penalty
        return fitness

    # √初始解构造算法,返回Machine_List
    def init_solution(self):
        release_time = self.release_time
        processing_time = self.processing_time
        weight = self.weight
        # rj非减排序并获得排序后的索引
        sorted_release_time_indexs = sorted(range(len(release_time)), key=lambda k: release_time[k])
        # rj非减排序并获得排序后的施放时间
        sorted_release_time = [release_time[i] for i in sorted_release_time_indexs]
        # rj非减排序并获得排序后的权重
        sorted_weight = [weight[i] for i in sorted_release_time_indexs]
        # 根据release_time的排序顺序对processing_time的列进行排序
        sorted_processing_time = [[row[i] for i in sorted_release_time_indexs] for row in processing_time]

        Machine_list = [[] for i in range(self.m)]
        WC = [0.0] * self.m  # 工件调度到不同机器上的WC的各种情况
        for j in range(self.n):
            Machine_list[0].append(sorted_release_time_indexs[j] + 1)  # 先从第一个机器开始
            WC[0] = self.calculate_WC(Machine_list)
            for i in range(1, self.m):  # 从第二个机器开始
                Machine_list[i - 1].pop()  # 上一个机器移除末尾工件
                Machine_list[i].append(sorted_release_time_indexs[j] + 1)
                WC[i] = self.calculate_WC(Machine_list)
            Machine_list[self.m - 1].pop()  # 上一个机器移除末尾工件
            # 返回最小元素值对应的机器索引
            i_index = WC.index(min(WC))
            Machine_list[i_index].append(sorted_release_time_indexs[j] + 1)
        return Machine_list

    # √初始种群,返回染色体集合XF  1+(NP-1)
    def init_population(self):
        NP = self.NP
        XF = [[] for i in range(NP)]
        for i in range(NP - 1):
            num_set = set()  # 查找元素是否存在的效率为O(1)
            while len(num_set) < self.n:
                num = np.random.uniform(1, self.m + 1)
                num = math.trunc(num * 1000) / 1000  # 值从1开始,截断保留三位小数
                if num not in num_set:
                    num_set.add(num)
            XF[i] = list(num_set)
        XF[NP - 1] = self.encode(self.init_solution())
        return XF

    # √选择 输入染色体种群XF，返回选择后的染色体种群select_XF
    def selection(self, XF):  # 二元锦标赛
        # 二元锦标赛
        NP = self.NP
        select_XF = []
        Fitness = []
        for xx in XF:
            xlist = self.decode(xx)
            xfit = self.calculate_fitness(xlist)
            Fitness.append(xfit)
        # 二元锦标赛（列表形式）
        for i in range(NP):
            A = np.random.choice(NP, size=2, replace=False)
            sample = []
            for a in A:
                fit_a = Fitness[a]
                a_fit = [a, fit_a]
                sample.append(a_fit)
            sortedsample = sorted(sample, key=lambda x: x[1], reverse=False)  # 从低到高排序
            newx_index = sortedsample[0][0]
            select_XF.append(XF[newx_index])
        return select_XF

    # √均匀交叉,输入染色体种群XF，根据例子给的Pc进行均匀交叉，输出交叉后的两个子代集合cross_X
    def crossover(self, XF):
        # 使用深拷贝复制输入的染色体种群 XF 到 Cro_XF，避免在操作过程中修改原始种群
        Cro_XF = copy.deepcopy(XF)
        NP = self.NP
        n = self.n
        Pc = self.Pc
        cross_X = []
        for i in range(int(NP / 2)):
            A = np.random.choice(NP, size=2, replace=False)
            A = A.tolist()
            P1 = Cro_XF[A[0]]  # 父代1
            P2 = Cro_XF[A[1]]  # 父代2
            child1 = [0 for k in range(n)]  # 子代1
            child2 = [0 for k in range(n)]  # 子代2
            # 均匀交叉
            for j in range(n):
                val = random.random()
                if val < Pc:
                    child1[j], child2[j] = P2[j], P1[j]
                else:
                    child1[j], child2[j] = P1[j], P2[j]
            cross_X.append(child1)
            cross_X.append(child2)
        return cross_X

    # √均匀变异,输入染色体种群XF，根据例子给的Pm进行变异，输出变异后的染色体种群Mut_XF
    def mutation(self, XF):
        Mut_XF = copy.deepcopy(XF)
        Pm = self.Pm
        NP = self.NP
        n = self.n
        m = self.m
        for i in range(NP):
            P1 = Mut_XF[i]  # 父代1
            for j in range(n):
                val = random.random()
                if val < Pm:
                    num = np.random.uniform(1, self.m + 1)
                    num = math.trunc(num * 1000) / 1000  # 值从1开始,截断保留三位小数
                    P1[j] = num
            Mut_XF[i] = P1
        return Mut_XF

    # √进行一次基于同一机器上相邻交换的局部搜索增强算法性质1，2.输入染色体个体X，输出增强后的染色体个体ls_X
    def onetime_local_search(self, X):
        m = self.m
        processing_time = self.processing_time
        release_time = self.release_time
        weight = self.weight
        XX = copy.deepcopy(X)  # 要进行局部搜索的染色体个体
        machinelist = self.decode(XX)
        Cj = self.AllJob_CompletionTime(machinelist)
        # 进行一次局优 k-1、k、k+1
        for i in range(m):
            for k in range(len(machinelist[i]) - 1):
                if k == 0:  # 即k为最开始的索引位置“0”处
                    C_pre_k = 0
                else:
                    C_pre_k = Cj[machinelist[i][k - 1] - 1]
                job1 = machinelist[i][k] - 1  # k位置的工件索引(值-1)
                job2 = machinelist[i][k + 1] - 1
                f1 = XX[job1] - int(XX[job1])  # 保留小数部分
                f2 = XX[job2] - int(XX[job2])
                w1 = weight[job1]
                w2 = weight[job2]
                p1 = processing_time[i][job1]
                p2 = processing_time[i][job2]
                r1 = release_time[job1]
                r2 = release_time[job2]
                if (p2 * w1 - p1 * w2 < 0 and C_pre_k >= max(r1, r2)) or (p2 * w1 - p1 * w2 < 0 and r2 <= r1):  # 性质1，2
                    machinelist[i][k] = job2 + 1
                    machinelist[i][k + 1] = job1 + 1  # 交换位置
                    Cj = self.AllJob_CompletionTime(machinelist)  # 更新Cj
                    XX[job1] = i + 1 + f2
                    XX[job2] = i + 1 + f1
                else:
                    pass
        return XX

    # √重复进行局部搜索直至无法提升.输入染色体个体X，输出增强后的染色体个体ls_X
    def multitime_local_search(self, X):
        XX1 = copy.deepcopy(X)
        XX2 = self.onetime_local_search(XX1)
        fitness_XX1 = self.calculate_fitness(self.decode(XX1))
        fitness_XX2 = self.calculate_fitness(self.decode(XX2))
        while fitness_XX2 < fitness_XX1:
            XX1 = copy.deepcopy(XX2)
            fitness_XX1 = fitness_XX2
            XX2 = self.onetime_local_search(XX1)
            fitness_XX2 = self.calculate_fitness(self.decode(XX2))
        return XX2

    # 基于贪婪策略的快速修复算法（优先处理资源消耗最大的工件）
    def repair_insert(self, X):
        m = self.m
        n = self.n
        U_total = self.calculate_U_total()
        processing_time = self.processing_time
        li = self.li
        Machine_list = self.decode(X)
        TC = self.calculate_TC(Machine_list)

        max_iterations = 100000
        iteration_count = 0

        while TC > U_total and iteration_count < max_iterations:
            iteration_count += 1
            # 计算每个工件的资源消耗
            job_resources = []
            for j in range(n):
                current_machine = int(X[j]) - 1
                resource = li[current_machine] * processing_time[current_machine][j]
                job_resources.append((j, resource, current_machine))

            # 按资源消耗降序排序，优先处理消耗大的工件
            job_resources.sort(key=lambda x: x[1], reverse=True)

            improved = False
            for j, _, current_machine in job_resources:
                # 寻找资源消耗更少的机器
                better_machines = []
                current_resource = li[current_machine] * processing_time[current_machine][j]

                for i in range(m):
                    if i == current_machine:
                        continue
                    new_resource = li[i] * processing_time[i][j]
                    if new_resource < current_resource:
                        # 计算移动后的收益（资源减少量）
                        gain = current_resource - new_resource
                        better_machines.append((i, gain))

                if better_machines:
                    # 选择收益最大的机器
                    better_machines.sort(key=lambda x: x[1], reverse=True)
                    target_machine = better_machines[0][0]

                    # 从当前机器移除工件
                    Machine_list[current_machine] = [x for x in Machine_list[current_machine] if x != j + 1]

                    # 插入到目标机器的最优位置（简化版：只尝试首尾位置）
                    if not Machine_list[target_machine]:
                        Machine_list[target_machine].append(j + 1)
                    else:
                        # 比较插入到开头和结尾的效果
                        head_list = [j + 1] + Machine_list[target_machine]
                        head_wc = self.calculate_one_machine_WC(target_machine + 1, head_list)

                        tail_list = Machine_list[target_machine] + [j + 1]
                        tail_wc = self.calculate_one_machine_WC(target_machine + 1, tail_list)

                        if head_wc <= tail_wc:
                            Machine_list[target_machine] = head_list
                        else:
                            Machine_list[target_machine] = tail_list

                    # 更新编码和TC
                    X = self.encode(Machine_list)
                    TC = self.calculate_TC(Machine_list)
                    improved = True
                    break  # 每次只移动一个工件就重新检查

            if not improved:
                break  # 无法继续改进
        return self.encode(Machine_list)

    # !!!!√随机修复算法,输入不合理的染色体解个体X，输出修复后的解repair_X
    # 若循环1000次仍不出结果，可能进入死循环，输出元素全为0的错误提示解repair_X
    def repair_random(self, X):
        m = self.m
        n = self.n
        U_total = self.calculate_U_total()
        processing_time = self.processing_time
        li = self.li
        Machine_list = self.decode(X)
        TC = self.calculate_TC(Machine_list)
        judge = 0
        while TC > U_total:
            if judge > 10000:
                break
            judge += 1
            J_set = [j for j in range(n)]  # 初始化工件集
            while True:
                # print(judge)
                # judge += 1
                j = random.choice(J_set)  # 随机选取一个工件j索引
                M_Jj = []  # 记录工件j可以在哪些机器上加工且会降低资源消耗
                Ma = int(X[j])  # 加工Jj的原机器值
                for i in range(m):  # 生成M_Jj
                    if li[Ma - 1] * processing_time[Ma - 1][j] > li[i] * processing_time[i][j]:
                        M_Jj.append(i + 1)  # 记录可行的机器值
                if len(M_Jj) > 0:  # 若可行机器集非空
                    break
                else:
                    J_set = [x for x in J_set if x != j]  # 若M_Jj为空，则从工件集中移除该工件j
            # 先把Ma上的该工件j移除
            Machine_list[Ma - 1] = [x for x in Machine_list[Ma - 1] if x != j + 1]
            # 再随机选取M_Jj中的机器值Mb
            Mb = random.choice(M_Jj)
            if len(Machine_list[Mb - 1]) == 0:
                best_Mb = Machine_list[Mb - 1][:0] + [j + 1] + Machine_list[Mb - 1][0:]  # 从位置0开始插入
            elif len(Machine_list[Mb - 1]) > 0:
                insert_index = random.randint(0, len(Machine_list[Mb - 1]))  # 随机选取插入位置,从0到数组长度
                best_Mb = Machine_list[Mb - 1][:insert_index] + [j + 1] + Machine_list[Mb - 1][insert_index:]
            Machine_list[Mb - 1] = best_Mb
            TC = self.calculate_TC(Machine_list)
        if judge > 10000:
            repair_X = [0 for i in range(n)]
        else:
            repair_X = self.encode(Machine_list)
        return repair_X

    # √RVNS中的扰动邻域.输入染色体解个体X，输出扰动后的解shake_X
    def shake(self, X):
        n = self.n
        Machine_list = self.decode(X)
        A = np.random.choice(n, size=2, replace=False)  # 随机选择两个工件
        A = A.tolist()
        J1 = A[0] + 1
        J2 = A[1] + 1  # 被选择的工件值
        M1 = int(X[A[0]])
        M2 = int(X[A[1]])  # 被选择的工件对应的机器值
        # print("在机器", M1, "上的工件", J1, "和在机器", M2, "上的工件", J2, "进行交换")
        if M1 != M2:
            Machine_list[M1 - 1] = [J2 if x == J1 else x for x in Machine_list[M1 - 1]]
            Machine_list[M2 - 1] = [J1 if x == J2 else x for x in Machine_list[M2 - 1]]  # 交换两工件
        else:
            for k in range(len(Machine_list[M1 - 1])):  # 获取两工件对应的索引
                if Machine_list[M1 - 1][k] == J1:
                    index1 = k
                if Machine_list[M1 - 1][k] == J2:
                    index2 = k
            if index1 != -1 and index2 != -1:
                Machine_list[M1 - 1][index1], Machine_list[M1 - 1][index2] = Machine_list[M1 - 1][index2], \
                                                                             Machine_list[M1 - 1][index1]
        shake_X = self.encode(Machine_list)
        return shake_X

    def rvnd1(self, X):
        m = self.m
        Machine_list = copy.deepcopy(self.decode(X))

        # 计算每台机器的总加权完工时间和
        machine_WC = [self.calculate_one_machine_WC(i + 1, Machine_list[i]) for i in range(m)]

        # 选择总加权完工时间和最大的机器
        max_WC_machine_index = machine_WC.index(max(machine_WC))
        original_one_WC = machine_WC[max_WC_machine_index]
        best_one_WC = original_one_WC
        best_Machine_list = Machine_list

        if len(Machine_list[max_WC_machine_index]) > 1:
            for j in range(len(Machine_list[max_WC_machine_index]) - 1):
                for k in range(j + 1, len(Machine_list[max_WC_machine_index])):
                    Machine_list_temp = copy.deepcopy(Machine_list)
                    Machine_list_temp[max_WC_machine_index][j], Machine_list_temp[max_WC_machine_index][k] = \
                        Machine_list_temp[max_WC_machine_index][k], \
                        Machine_list_temp[max_WC_machine_index][j]
                    new_one_WC = self.calculate_one_machine_WC(max_WC_machine_index + 1,
                                                               Machine_list_temp[max_WC_machine_index])
                    if new_one_WC < best_one_WC:
                        best_one_WC = new_one_WC
                        best_Machine_list = Machine_list_temp
                        return self.encode(best_Machine_list)

        return self.encode(best_Machine_list)

    # 选择总加权完工时间和最大的机器，直至下降
    def rvnd2(self, X):
        m = self.m
        machine_list = copy.deepcopy(self.decode(X))

        # 计算每台机器的总加权完工时间和
        machine_WC = [self.calculate_one_machine_WC(i + 1, machine_list[i]) for i in range(m)]

        # 选择总加权完工时间和最大的机器
        max_WC_machine_index = machine_WC.index(max(machine_WC))
        original_one_WC = machine_WC[max_WC_machine_index]
        best_one_WC = original_one_WC
        best_machine_list = machine_list

        for j in range(len(machine_list[max_WC_machine_index])):
            job = machine_list[max_WC_machine_index][j]
            for insert_index in range(len(machine_list[max_WC_machine_index])):
                if insert_index != j:
                    machine_list_temp = copy.deepcopy(machine_list)
                    del machine_list_temp[max_WC_machine_index][j]
                    machine_list_temp[max_WC_machine_index].insert(insert_index, job)
                    new_one_WC = self.calculate_one_machine_WC(max_WC_machine_index + 1,
                                                               machine_list_temp[max_WC_machine_index])
                    if new_one_WC < best_one_WC:
                        best_one_WC = new_one_WC
                        best_machine_list = machine_list_temp
                        return self.encode(best_machine_list)

        return self.encode(best_machine_list)

    # 选择总加权完工时间和最大的机器和另一台随机选择的不同机器，若交换后这两台机器提供的总加权完工时间和的和下降则停止搜索输出该解
    def rvnd3(self, X):
        m = self.m
        Machine_list = copy.deepcopy(self.decode(X))

        # 计算每台机器的总加权完工时间和
        machine_WC = [self.calculate_one_machine_WC(i + 1, Machine_list[i]) for i in range(m)]

        # 选择总加权完工时间和最大的机器
        max_WC_machine_index = machine_WC.index(max(machine_WC))

        # 随机选择另一台不同的机器
        other_machines = [i for i in range(m) if i != max_WC_machine_index]
        if not other_machines:  # 防止只有一台机器的情况
            return X
        other_machine_index = random.choice(other_machines)

        # 获取两台机器上的工件列表
        machine_i_jobs = Machine_list[max_WC_machine_index]
        machine_h_jobs = Machine_list[other_machine_index]

        # 处理空机器情况
        if not machine_i_jobs or not machine_h_jobs:
            return X

        # 原始总加权完工时间和
        original_one_WC_i = machine_WC[max_WC_machine_index]
        original_one_WC_h = machine_WC[other_machine_index]
        original_total_WC = original_one_WC_i + original_one_WC_h

        # # 计算搜索空间大小和停止阈值（搜索空间的一半）
        # search_space = len(machine_i_jobs) * len(machine_h_jobs)
        # stop_threshold = (search_space + 1) // 2  # 向上取整，确保至少搜索一半
        search_count = 0

        # 生成所有可能的工件对索引并随机打乱，增加早期找到改进解的概率
        job_pairs = [(j, k) for j in range(len(machine_i_jobs))
                     for k in range(len(machine_h_jobs))]
        random.shuffle(job_pairs)

        # 遍历工件对进行交换尝试
        for j, k in job_pairs:
            search_count += 1

            # 交换工件
            Machine_list_temp = copy.deepcopy(Machine_list)
            Machine_list_temp[max_WC_machine_index][j], Machine_list_temp[other_machine_index][k] = \
                Machine_list_temp[other_machine_index][k], Machine_list_temp[max_WC_machine_index][j]

            # 计算新的加权完工时间和
            new_one_WC_i = self.calculate_one_machine_WC(max_WC_machine_index + 1,
                                                         Machine_list_temp[max_WC_machine_index])
            new_one_WC_h = self.calculate_one_machine_WC(other_machine_index + 1,
                                                         Machine_list_temp[other_machine_index])
            new_total_WC = new_one_WC_i + new_one_WC_h

            # 找到改进解，立即返回
            if new_total_WC < original_total_WC:
                return self.encode(Machine_list_temp)

            # 搜索超过一定次数仍无改进，停止搜索
            if search_count >= min(5, math.ceil(self.n / self.m)):
                break

        # 未找到改进解，返回原解
        return X

    # 选择总加权完工时间和最大的机器和另一台随机选择的不同机器，若插入后这两台机器提供的总加权完工时间和的和下降则停止搜索输出该解
    def rvnd4(self, X):
        m = self.m
        Machine_list = copy.deepcopy(self.decode(X))

        # 计算每台机器的总加权完工时间和
        machine_WC = [self.calculate_one_machine_WC(i + 1, Machine_list[i]) for i in range(m)]

        # 选择总加权完工时间和最大的机器
        max_WC_machine_index = machine_WC.index(max(machine_WC))

        # 随机选择另一台不同的机器
        other_machines = [i for i in range(m) if i != max_WC_machine_index]
        if not other_machines:  # 防止只有一台机器的情况
            return X
        other_machine_index = random.choice(other_machines)

        # 获取两台机器上的工件列表
        source_machine_jobs = Machine_list[max_WC_machine_index]
        target_machine_jobs = Machine_list[other_machine_index]

        # 处理源机器为空的情况
        if not source_machine_jobs:
            return X

        # 原始总加权完工时间和
        original_one_WC_i = machine_WC[max_WC_machine_index]
        original_one_WC_h = machine_WC[other_machine_index]
        original_total_WC = original_one_WC_i + original_one_WC_h

        # # 计算搜索空间大小和停止阈值（搜索空间的一半）
        job_count = len(source_machine_jobs)
        insert_pos_count = len(target_machine_jobs) + 1  # 插入位置比工件数多1
        # search_space = job_count * insert_pos_count
        # stop_threshold = (search_space + 1) // 2  # 向上取整
        search_count = 0

        # 生成所有可能的(工件, 插入位置)对并随机打乱
        search_candidates = [
            (j, pos)
            for j in range(job_count)
            for pos in range(insert_pos_count)
        ]
        random.shuffle(search_candidates)

        # 遍历候选进行插入尝试
        for j, insert_index in search_candidates:
            search_count += 1

            # 执行插入操作
            Machine_list_temp = copy.deepcopy(Machine_list)
            # 从源机器移除工件
            job = Machine_list_temp[max_WC_machine_index].pop(j)
            # 插入到目标机器
            Machine_list_temp[other_machine_index].insert(insert_index, job)

            # 计算新的加权完工时间和
            new_one_WC_i = self.calculate_one_machine_WC(
                max_WC_machine_index + 1,
                Machine_list_temp[max_WC_machine_index]
            )
            new_one_WC_h = self.calculate_one_machine_WC(
                other_machine_index + 1,
                Machine_list_temp[other_machine_index]
            )
            new_total_WC = new_one_WC_i + new_one_WC_h

            # 找到改进解，立即返回
            if new_total_WC < original_total_WC:
                return self.encode(Machine_list_temp)

            # 搜索超过一定次数仍无改进，停止搜索
            if search_count >= min(5, math.ceil(self.n / self.m)):
                break

        # 未找到改进解，返回原解
        return X

    # √RVNS中的局部搜索邻域.输入染色体解个体X和邻域序号k，1为同一机器交换；2为同一机器插入；3为不同机器交换；4为不同机器插入。
    # 输出局部搜索后的解rvnd_X，（不包括X）
    def rvnd(self, X, k):
        n = self.n
        X_rvnd = copy.deepcopy(X)
        if k == 1:
            return self.rvnd1(X_rvnd)
        elif k == 2:
            return self.rvnd2(X_rvnd)
        elif k == 3:
            return self.rvnd3(X_rvnd)
        elif k == 4:
            return self.rvnd4(X_rvnd)

    # √寻找种群最优个体,输入种群XF，输出最优个体X_best
    def find_best(self, XF):
        best_fitness = float('inf')  # 初始化为正无穷大
        X_best = None
        for X in XF:
            Machine_list = self.decode(X)
            fitness = self.calculate_fitness(Machine_list)
            if fitness < best_fitness:
                best_fitness = fitness
                X_best = X
        return X_best

    # IFGA算法,返回最优解S_best
    def IFGA_def(self):

        t0 = time.time()
        ii = 0  # 迭代次数，防止和“i”命名冲突
        Iter = self.Iter  # 最大迭代次数
        count = 0  # 未提升最优解次数
        # 初始化种群XF(0)
        XF = self.init_population()
        S_best = self.find_best(XF)  # 记录最优解
        S_best_index = XF.index(S_best)  # 记录最优解对应的索引
        TC_best = self.calculate_TC(self.decode(S_best))  # 记录最优解的资源消耗
        U_total = self.calculate_U_total()  # 记录资源消耗阈值

        # 比较S1和S2的适应度，更新最优解
        if TC_best > U_total:
            S1 = self.repair_insert(S_best)
            S2 = self.repair_random(S_best)
            # 比较S1和S2的适应度，更新最优解
            if sum(S2) < 1:  # 随机修复超次数
                S_best = copy.deepcopy(S1)
                print("随机修复超次数")
            else:
                if self.calculate_fitness(self.decode(S1)) <= self.calculate_fitness(self.decode(S2)):
                    S_best = copy.deepcopy(S1)
                else:
                    S_best = copy.deepcopy(S2)

        # 利用性质增强S_best的解质量
        S_best = self.onetime_local_search(S_best)
        # 计算S_best的适应度值
        fitness_S_best = self.calculate_fitness(self.decode(S_best))
        elite_S = copy.deepcopy(S_best)  # 保留精英个体
        XF[S_best_index] = S_best
        t1 = time.time()
        # print("初始化种群时间：", t1 - t0)

        while ii < Iter:
            # GA
            # 选择操作
            t2 = time.time()
            select_XF = self.selection(XF)
            t3 = time.time()
            # print("选择操作时间：", t3 - t2)
            # 交叉操作
            cross_X = self.crossover(select_XF)
            t4 = time.time()
            # print("交叉操作时间：", t4 - t3)
            # 变异操作
            XF1 = self.mutation(cross_X)
            t5 = time.time()
            # print("变异操作时间：", t5 - t4)
            # 合并XF(i)、XF1和elite_S
            combined_population = XF + XF1 + [elite_S]
            # 计算每个个体的适应度
            fitness_values = []
            for individual in combined_population:
                machine_list = self.decode(individual)
                fitness = self.calculate_fitness(machine_list)
                fitness_values.append(fitness)
            # 根据适应度非减排序
            sorted_indices = sorted(range(len(combined_population)), key=lambda k: fitness_values[k])
            sorted_population = [combined_population[idx] for idx in sorted_indices]
            # 选择前NP个个体组成XF(i+1)
            XF = sorted_population[:self.NP]
            # 选择XF(i+1)的最优个体记为S_XF_best
            S_XF_best = copy.deepcopy(XF[0])
            # 利用性质增强S_XF_best的解质量
            S_XF_best = self.onetime_local_search(S_XF_best)
            t6 = time.time()
            # print("增强最优解时间：", t6 - t5)
            # 计算最优个体S_XF_best的适应度
            # 由于 sorted_indices 是按适应度排序的索引，所以第一个索引对应的适应度就是 S_XF_best 的适应度
            fitness_S_XF_best = fitness_values[sorted_indices[0]]
            # 记录最优个体S_XF_best在XF中的索引S_XF_best_index
            S_XF_best_index = 0
            t7 = time.time()
            # print("VNS之前的操作时间：", t7 - t6)
            # VNS
            N_ls = [1, 2, 3, 4]  # 局部搜索邻域

            # N_ls = [random.choice(N_ls)]
            S_VNS_best = copy.deepcopy(S_XF_best)
            # 计算初始最优个体S_VNS_best的适应度
            fitness_S_VNS_best = fitness_S_XF_best
            # 由XF的最优解S_XF_best经过VNS产生新解S_VNS_best
            while N_ls:
                # 对解S_VNS_best进行扰动
                S_shake = self.shake(S_VNS_best)
                # 从N_ls中随机选择邻域进行局部搜索
                Nk = random.choice(N_ls)
                S_ls = self.rvnd(S_shake, Nk)
                # 利用性质提高解质量
                S_ls = self.onetime_local_search(S_ls)
                # 计算扰动后解的适应度
                fitness_S_ls = self.calculate_fitness(self.decode(S_ls))
                # 比较S_VNS_best和S_ls的适应度
                if fitness_S_ls < fitness_S_VNS_best:
                    S_VNS_best = copy.deepcopy(S_ls)
                    fitness_S_VNS_best = fitness_S_ls
                    N_ls = [1, 2, 3, 4]  # 重置邻域
                else:
                    N_ls.remove(Nk)  # 移除已使用的邻域
            t8 = time.time()
            # print("VNS操作时间：", t8 - t7)
            count += 1
            if fitness_S_VNS_best < fitness_S_best:
                S_best = copy.deepcopy(S_VNS_best)
                fitness_S_best = fitness_S_VNS_best
                S_XF_best = copy.deepcopy(S_VNS_best)
                elite_S = copy.deepcopy(S_VNS_best)
                count = 0
            elif fitness_S_VNS_best < fitness_S_XF_best:
                S_XF_best = copy.deepcopy(S_VNS_best)
                elite_S = copy.deepcopy(S_VNS_best)
            else:
                elite_S = copy.deepcopy(S_XF_best)

            # 将更新后的S_XF_best放回XF中
            XF[S_XF_best_index] = S_XF_best

            # 若未提升最优解S_best达到K次，按比例干扰种群XF
            # 随机生成规模为math.floor(NP*10%)的新个体，与原种群XF(i+1)合并，根据适应度非减排序选择前NP个个体组成种群XF(i+1),count←0;
            if count >= self.K:
                # 随机生成规模为math.floor(NP*10%)的新个体
                new_XF_size = math.floor(self.NP * 0.1)
                new_XF = [[] for i in range(new_XF_size)]
                for i in range(new_XF_size):
                    num_set = set()  # 查找元素是否存在的效率为O(1)
                    while len(num_set) < self.n:
                        num = np.random.uniform(1, self.m + 1)
                        num = math.trunc(num * 1000) / 1000  # 值从1开始,截断保留三位小数
                        if num not in num_set:
                            num_set.add(num)
                    new_XF[i] = list(num_set)
                # 与原种群XF(i+1)合并
                combined_population_shake = XF + new_XF
                # 计算每个个体的适应度
                fitness_values_shake = []
                for individual in combined_population_shake:
                    machine_list = self.decode(individual)
                    fitness = self.calculate_fitness(machine_list)
                    fitness_values_shake.append(fitness)

                # 根据适应度非减排序
                sorted_indices_shake = sorted(range(len(combined_population_shake)),
                                              key=lambda k: fitness_values_shake[k])
                sorted_population_shake = [combined_population_shake[idx] for idx in sorted_indices_shake]

                # 选择前NP个个体组成种群XF(i+1)
                XF = sorted_population_shake[:self.NP]

                count = 0  # 重置计数器
            ii += 1
        TC_best = self.calculate_TC(self.decode(S_best))  # 记录最优解的资源消耗
        # 对不合理的解进行修复
        if TC_best > U_total:
            S1 = self.repair_insert(S_best)
            S2 = self.repair_random(S_best)
            # 比较S1和S2的适应度，更新最优解
            if sum(S2) < 1:  # 随机修复超次数
                S_best = S1
                print("随机修复超次数")
            else:
                if self.calculate_fitness(self.decode(S1)) <= self.calculate_fitness(self.decode(S2)):
                    S_best = S1
                else:
                    S_best = S2
        # 利用性质增强S_best的解质量
        S_best = self.onetime_local_search(S_best)
        return S_best
