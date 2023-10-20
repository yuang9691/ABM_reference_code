'''
与原模型相比，增加了对负面口碑的处理
'''
from copy import deepcopy
import numpy as np
import networkx as nx
import time
import multiprocessing
import matplotlib.pyplot as plt

NUM_CORNS = multiprocessing.cpu_count()  # cpu核数

def cal_npv(a_array, rate=0.1):
    return np.sum([x * (1 + rate) ** -i for i, x in enumerate(a_array)])

def page_rank_centrality(g, alpha=0.85, max_iter=100, epsilon=1e-6):
    if len(g) == 0:
        return {}
    
    if not nx.is_directed(g):
        g = g.to_directed()
    
    # 1. 初始化pr值
    page_rank = dict.fromkeys(g.nodes(), 1 / g.number_of_nodes())
    beta = (1 - alpha) / g.number_of_nodes()  # 防止节点pr值为0     
    in_degree_dict = {i: max(1, v) for i, v in g.in_degree()} # 为了计算，将所有入度为0的节点的值调整为1
    # 2. 迭代
    flag = 1
    while flag <= max_iter:
        change = 0
        for i in g:  # 更新所有节点的pr值
            pr = alpha * np.sum([page_rank[j] / in_degree_dict[j] for j in g.successors(i)]) + beta
            change += abs(page_rank[i] - pr)
            page_rank[i] = pr
        
        if change <= epsilon:
            print(f'迭代在{flag}轮后达到阈值，终止')
            break

        flag += 1
    else:
        print(f'迭代在{max_iter}轮后终止')

    return list(page_rank.items())

def transfer_data(x):
    '''
    把不规则形状的二维数据转换为矩阵
    '''
    max_len = max([len(u) for u in x])
    new_x = np.zeros((len(x), max_len))
    for i in range(len(x)):
        for j in range(len(x[i])):
            new_x[i, j] = x[i][j]
            
    return new_x


class InfluencerDiffusionModel:
    """
    信息竞争情景下的创新扩散
    """

    def __init__(
        self, G, omega, rho, mu, c=0.7, gamma=0.01, sigma=0.2, seed_strategy="celebrities", measure="outDegree"
    ):
        self.G = G.copy()
        self.seed_strategy = seed_strategy
        self.omega = omega  # fraction of network willing to pay [0, 1, 2, 3, 4, 5] 10
        self.rho = rho  # brand's influencer hiring investment limit [0.1, 0.2]
        self.gamma = gamma  # probability of agent being a disappointer
        self.mu = mu  # the mean of initial adoption intention
        self.sigma = sigma  # the std of initial adoption intention
        self.c = c  # increment of intention update
        
        max_outdegree = max(self.G.out_degree, key=lambda x: x[1])[1]  # 最大出度
        for i in self.G:
            # hiring cost of agent i
            self.G.nodes[i]["v"] = 100*self.G.out_degree(i)/max_outdegree  # 标准化出度
            self.G.nodes[i]["hiring_cost"] = 0.01 * self.G.out_degree(i)
            # 是否为潜在的不满意节点：如果不满意，则向邻居发送负面口碑信息
            if np.random.rand() < self.gamma:
                self.G.nodes[i]["is_disappointed"] = True
            else:
                self.G.nodes[i]["is_disappointed"] = False
        
        self.set_node_type(measure=measure)
        self.eta, self.seeds = self.choose_seeds(seed_strategy)
        
    def set_node_type(self, measure="outDegree"):
        if measure == "outDegree":
            sorted_nodes = sorted(self.G.nodes, key=lambda x: self.G.out_degree(x))
        elif measure == "pageRank":
            sorted_nodes = sorted(self.G.nodes, key=lambda x: self.G.nodes[x]["page_rank_centrality"])
        else:
            raise(ValueError, "Please input correct measure!")
        
        for i in range(self.G.number_of_nodes()):
            j = sorted_nodes[i]
            if i < int(self.G.number_of_nodes()*0.6):
                self.G.nodes[j]["type"] = "followers"
            elif i < int(self.G.number_of_nodes()*0.7):
                self.G.nodes[j]["type"] = "micro_influencers"
            elif i < int(self.G.number_of_nodes()*0.8):
                self.G.nodes[j]["type"] = "mid_tier_influencers"
            elif i < int(self.G.number_of_nodes()*0.9):
                self.G.nodes[j]["type"] = "macro_influencers"
            elif i < int(self.G.number_of_nodes()*0.975):
                self.G.nodes[j]["type"] = "mega_influencers"
            else:
                self.G.nodes[j]["type"] = "celebrities"
            

    def choose_seeds(self, seed_strategy="celebrities"):
        """
        Choose seeds according to the given seeding strategy and the total hiring cost.
        """
        candidate_seeds, seeds = [], []
        for i in self.G:
            if self.G.nodes[i]["type"] == seed_strategy:
                candidate_seeds.append(i)
        
        if len(candidate_seeds) == 0:
            raise ValueError(f"不存在该种子策略{seed_strategy}!")
                    
        total_hiring_cost = deepcopy(self.rho)
        # 通过控制总成本确定种子数量
        eta = 0  # 总成本
        np.random.shuffle(candidate_seeds)
        for i in candidate_seeds:
            cost = self.G.nodes[i]['hiring_cost']
            if eta + cost <= total_hiring_cost:
                eta += cost
                seeds.append(i)
            else:
                break
        
        return eta, seeds # 种子节点集合

    def init_simulation_params(self):
        """
        初始化每一次模拟的随机参数
        Initiate properites changed across simulations.
        """
        for i in self.G:
            # probability agent i interested in product
            u = 0
            while True:  # 截尾正态分布[0, 1]
                a = self.mu + np.random.randn()*self.sigma
                if 0 <= a <= 1:
                    self.G.nodes[i]["adoptIntention"] = a
                    break
                
                u += 1
                if u > 100:  # 防止陷入无穷循环
                    self.G.nodes[i]["adoptIntention"] = self.mu
                    print(f"node: {i}, adoptIntention设置超过次数100! 取均值!")
                    break
                
            # probability of agent i being active
            self.G.nodes[i]["activeness"] = np.random.rand()
            self.G.nodes[i]["willing"] = 1 if self.omega > np.random.rand() else 0  # 是否为潜在顾客
            self.G.nodes[i]["threshold"] = np.random.rand()  # 采纳阈值
            
            if i in self.seeds:
                self.G.nodes[i]["state"] = 1  # 0: 未决定状态, 1: 购买状态
                self.G.nodes[i]["reach"] = 1  # 是否在知晓产品，advertized
            else:
                self.G.nodes[i]["state"] = 0
                self.G.nodes[i]["reach"] = 0  # 未知晓产品
                
        #  influence of agent i on agent j
        for edge in self.G.edges():
            self.G.edges[edge]["weight"] = np.random.rand()  # 边权
            
    def positive_interaction(self, i, j):
        """
        正面互动: i -> j
        """
        self.G.nodes[j]["adoptIntention"] += self.G.edges[(i, j)]["weight"]*self.c
            
    def negative_interaction(self, i, j):
        """
        负面交互: i -> j
        """
        self.G.nodes[j]["adoptIntention"] -= self.G.edges[(i, j)]["weight"]*self.c*2
        for k in self.G.successors(j):  # 负面口碑传播2个度，影响减半
            if self.G.nodes[k]["state"] == 0:
                self.G.nodes[k]["adoptIntention"] -= self.G.edges[(j, k)]["weight"]*self.c
    
    def purchase_decision(self, i):
        """
        purchase decision of agent j influenced by agent i
        阈值规则：如果采纳意愿大于阈值，则购买
        """
        if self.G.nodes[i]["adoptIntention"] > self.G.nodes[i]["threshold"]:
            if self.G.nodes[i]["is_disappointed"]:   # 是否为失望者
                self.G.nodes[i]["state"] = -1
            else:
                self.G.nodes[i]["state"] = 1             

    def diffuse(self):
        """
        一次模拟过程
        """
        self.init_simulation_params()  # 初始化
        num_reach = []  # 接受到广告的用户
        num_adopters = []  # 购买顾客
        num_disappointer = []  # 不满意客户
        influencers = deepcopy(self.seeds)
        runs = 0
        while True:
            next_influencers = []
            adopters = 0
            disappointers = 0
            reach = 0
            for i in influencers:
                for j in self.G.successors(i):
                    # 如果j的状态为已采纳或j无意愿采纳，则不评估j的采纳意向
                    if self.G.nodes[j]["state"] != 0 or self.G.nodes[j]["willing"] == 0:
                        continue
                    
                    if self.G.nodes[j]["reach"] == 0:  # 防止重复计算
                        self.G.nodes[j]["reach"] = 1
                        reach += 1
                    
                    # 如果j处于激活状态, j更新采纳意向，执行购买决策
                    if self.G.nodes[j]["activeness"] > np.random.rand():
                        # (1) i, j互动
                        if self.G.nodes[i]["state"] == 1:  # 正面影响
                            self.positive_interaction(i, j)
                        elif self.G.nodes[i]["state"] == -1:  # 负面影响
                            self.negative_interaction(i, j)
                        else:  # 无影响
                            pass
                        
                        # (2) j进行购买决策
                        self.purchase_decision(j)
                        if self.G.nodes[j]["state"] == 1:
                            adopters += 1
                            next_influencers.append(j)  # 加入下一轮的influencers行列
                        elif self.G.nodes[j]["state"] == -1:
                            disappointers += 1
                            next_influencers.append(j)  # 加入下一轮的influencers行列
                        else:
                            pass
                            
            num_adopters.append(adopters)
            num_disappointer.append(disappointers)
            num_reach.append(reach)
            
            runs += 1
            if runs > 50:
                print("时间步超过50!停止模拟!")
                break
            
            if next_influencers:
                influencers = next_influencers
            else:  # 如果下一轮没有影响者，则终止循环
                break

        return num_reach, num_adopters, num_disappointer

    def multi_diffuse(self, num_runs=10):
        """
        多次模拟过程
        """
        reach_cont = []
        adopter_cont = []
        disappointer_cont = []

        if NUM_CORNS >= 2:
            pool = multiprocessing.Pool(processes=NUM_CORNS - 1)
            result = []
            for _ in range(num_runs):
                a = pool.apply_async(self.diffuse)
                result.append(a)

            pool.close()
            pool.join()
            for res in result:
                data = res.get()
                reach_cont.append(data[0])
                adopter_cont.append(data[1])
                disappointer_cont.append(data[2])
        else:
            for _ in range(num_runs):
                data = self.diffuse()
                reach_cont.append(data[0])
                adopter_cont.append(data[1])
                disappointer_cont.append(data[2])

        return reach_cont, adopter_cont, disappointer_cont
    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # G = nx.barabasi_albert_graph(10000, 6)
    # G = G.to_directed() if not nx.is_directed(G) else G
    G = nx.read_gpickle("empiricalNetworks/anybeat.gpickle")
    max_outdegree = sorted(G.out_degree, key=lambda x: x[1])[-1][1]
    rho = 2 * max_outdegree * 0.01  # hiring investment
    parameters = {
                  "G": G,  # social netowrk
                  "omega": 0.9,  # fraction of network willing to pay
                  "rho": rho,  # brand's influencer hiring investment
                  "mu": 0.1,  # the mean of influencers' initial adoption intention
                  "sigma": 0.1,  # the std of influencers' initial adoption intention
                  "c": 0.7,  # increment of adoption intention
                  "gamma": 0.01,  # probability of agents as a disappointer
                  "seed_strategy": "macro_influencers",  # influencer type: micro_influencers, mid_tier_influencers, macro_influencers, mega_influencers, celebrities
                  "measure": "outDegree"
    }
    
    t1 = time.perf_counter()
    abm = InfluencerDiffusionModel(**parameters)
    res = abm.multi_diffuse(num_runs=50)
    eta = abm.eta
    reach_cont = transfer_data(res[0])
    adopter_cont = transfer_data(res[1])
    disappointer_cont = transfer_data(res[2])
    
    mean_reach = np.mean(reach_cont, axis=0)
    mean_adopter = np.mean(adopter_cont, axis=0)
    mean_disappointer = np.mean(disappointer_cont, axis=0)
    psi = np.sum(mean_reach)
    chi = np.sum(mean_adopter)
    num_disappointers = np.sum(mean_disappointer)
    customer_acquisition_cost = eta/chi
    conversion_ratio = chi/psi
    npv = cal_npv(mean_adopter)
    print(f"time elasped: {time.perf_counter()-t1:.2f} s")
    print(f"influencer type: {abm.seed_strategy}, number of influencer: {len(abm.seeds)}")
    print(f"number of buyers: {chi:.4f}, number of reaches: {psi:.4f}")
    print(f"customer acquisition cost: {customer_acquisition_cost:.4f}\nconversion ratio:{conversion_ratio:.4f}")
    print(f"NPV of diffusion: {npv:.2f}")
    
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(len(mean_reach)), mean_reach, 'ko-', label="Reaches")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("#reaches")
    ax1.legend(loc="best")
       
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.arange(len(mean_adopter)), mean_adopter, 'r*-', label="Buyers")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("#adopters")
    ax2.legend(loc="best")
    plt.show()