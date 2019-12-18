"""
Gas Station Refueling example

Covers:

- Resources: Resource
- Resources: Container
- Waiting for other processes

Scenario:
  A gas station has a limited number of gas pumps that share a common
  fuel reservoir. Cars randomly arrive at the gas station, request one
  of the fuel pumps and start refueling from that reservoir.

  A gas station control process observes the gas station's fuel level
  and calls a tank truck for refueling if the station's level drops
  below a threshold.

"""

TRAINING_TIME = 10
CAPACITY = 150
SEG_SIZE = 20
g = 0.01

latest_n = 5 #取最近五个预测带宽取平均


import simpy
import importlib
import numpy as np
import os
import random
# import tensorflow as tf
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from server import Server
from simpy.events import AnyOf, AllOf
import heapq


RANDOM_SEED = 42
GAS_STATION_SIZE = 200     # liters
THRESHOLD = 10             # Threshold for calling the tank truck (in %)
FUEL_TANK_SIZE = 50        # liters
FUEL_TANK_LEVEL = [5, 25]  # Min/max levels of fuel tanks (in liters)
REFUELING_SPEED = 2        # liters / second
TANK_TRUCK_TIME = 300      # Seconds it takes the tank truck to arrive
T_INTER = [30, 300]        # Create a car every [min, max] seconds
SIM_TIME = 1000            # Simulation time in seconds

class client_simulate:
    def __init__(self, env,idx, num_clients,k, seg,server_num):
        self.exit_bw = simpy.Container(env, init=CAPACITY, capacity=CAPACITY)
        self.transfer_time = [env.now]
        self.training_time = [env.now]
        self.idx = idx
        self.seg_transfer_time = [0]*num_clients
        self.send_que = simpy.Container(env, init=0, capacity=1000)
        self.sigal = False
        self.last_que_num = 0
        self.is_que_change = False
        self.k = k
        self.max_seg_transfer_time = 0
        self.round_signal = False
        self.round = env.event()
        self.seg_num_count = 0
        self.seg = seg
        self.clients_num = num_clients
        self.server_num = server_num
        # self.round_signal.succeed()

    def train_process(self, env, my_round, num_clients, client_simulate_list, bandwidth):
        if my_round != 0:
            while not self.round_signal:
                yield env.timeout(0.01)
        self.round_signal = False

        print('-------------client [%d] round [%s] begin at %f:------------'%(self.idx,my_round,env.now))
        yield env.timeout(TRAINING_TIME)
        self.sigal = True
        self.training_time.append(env.now)
        idx_list = self.get_idx_list(num_clients, self.k)
        print('【Time:', env.now, '】', self.idx, 'pull from', idx_list)
        events = [env.process(self.get_transfer_time(env, client_simulate_list, bandwidth, my_round, i)) for i in idx_list]
        yield AllOf(env,events)
        print('【Time:', env.now,'】【Round: %s】【Id: %d】'%(my_round, self.idx), 'transfer')
        self.transfer_time.append(env.now)
        self.round_signal = True

    def choose_best_segment(self, replica):
        target = []
        # num = np.random.rand()
        # e = 0.5
        for i in range(replica):
            if self.server_num!= 1:
                a = np.random.randint(0, self.clients_num)
                while a == self.idx or a in target:
                    a = np.random.randint(0, self.clients_num)
                target.append(a)
            else:
                target.append(0)
        # if num < e:
        #     for i in range(replica):
        #         a = np.random.randint(0, self.clients_num)
        #         while a == self.idx or a in target:
        #             a = np.random.randint(0, self.clients_num)
        #         target.append(a)
        # else:
        #     pridict = []
        #     for bandwidth in self.pridict_bandwidth:
        #         if len(bandwidth) < latest_n:
        #             pridict.append(np.mean(bandwidth))
        #         else:
        #             pridict.append(np.mean(bandwidth[-latest_n:]))
        #     target = list(map(pridict.index, heapq.nlargest(latest_n, pridict)))
        return target

    def get_transfer_time(self, env, client_simulate_list, bandwidth, my_round,idx_list ):
        events = [env.process(self.pull_seg(env, client_simulate_list[i], i, bandwidth, my_round))for i in idx_list]
        yield AllOf(env, events)
        # print(self.seg_transfer_time, idx_list)
        self.max_seg_transfer_time = np.max(self.seg_transfer_time)

    def get_idx_list(self, client_num,k):
        idx_list = []
        for i in range(self.seg):
            target = self.choose_best_segment(self.k)
            idx_list.append(target)
        print(idx_list)
        return idx_list
        # client_num_list = list(range(client_num))
        # client_num_list.remove(self.idx)
        # return np.random.choice(client_num_list, size=k, replace=False, p=None)

    def pull_seg(self, env, client_simulate,idx, bandwidth, my_round):
        # called by other client
        # print('!!!【Time:', env.now,'】【Round: %s】【Id: %d】----segment-----【Id：%d】'%(my_round, self.idx,client_simulate.idx))
        while not client_simulate.sigal:
            yield env.timeout(0.01)
        # if my_round!= 0:
        # yield client_simulate.sigal
        yield client_simulate.send_que.put(1)
        exit_bw = client_simulate.exit_bw.level
        bottleneck_num = client_simulate.send_que.level
        link_bw = bandwidth[idx][self.idx]
        yield env.process(self.que_monitor(env, client_simulate, exit_bw, link_bw, bottleneck_num))
        yield client_simulate.send_que.get(1)
        # self.seg_transfer_time[idx] = num*0.1
        print('【Time:', env.now,'】【Round: %s】【Id: %d】-----pulling-----【Id：%d】'%(my_round, self.idx,client_simulate.idx))

    def que_monitor(self, env, client_simulate, exit_bw, link_bw,bottleneck_num):
        residual_seg_size = SEG_SIZE/self.seg
        final_bw = np.min([exit_bw/bottleneck_num, link_bw])
        last_que = bottleneck_num
        num = 0
        while residual_seg_size>=0:
            yield env.timeout(g)
            # yield env.timeout(0.1)
            residual_seg_size -= g * final_bw
            current_que = client_simulate.send_que.level
            # print(2222, current_que, last_que, self.idx, client_simulate.idx, env.now)
            if current_que != last_que:
                # print(1111111111, client_simulate.idx, current_que, last_que, final_bw, exit_bw, link_bw, env.now)
                last_que = current_que
                final_bw = np.min([exit_bw/current_que, link_bw])
            else:
                last_que = current_que
        # return num
        # print('【Time:', env.now, '】', self.idx, 'pull done')

class Server:
    def __init__(self,  clients_num):
        self.selected_clients = []
        self.updates = []
        self.clients_num = clients_num
        self.bandwidth = []
        self.transfer_time = []
        self.init_bandwidth()


    def init_bandwidth(self):
        for i in range(self.clients_num):
            a = []
            for j in range(self.clients_num):
                if i == j:
                    a.append(0)
                else:
                    a.append(np.random.randint(1, 100))
            self.bandwidth.append(a)

    def select_clients(self, my_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return self.select_clients

class set_up():
    def __init__(self, env):
        self.replica = 6
        self.segment = 1
        self.client_num = 6
        self.server_num = 1
        self.num_rounds = 50
        self.server = Server(self.client_num)
        self.clients = [client_simulate(env, i, self.client_num, self.replica, self.segment,self.server_num) for i in range(self.client_num)]
        self.main_proc = env.process(self.main(env))

    def main(self,env):
        for i in range(self.num_rounds):
            yield env.process(self.round(env, self.client_num, self.clients, self.server.bandwidth, i))
            # self.round_signal = False

    def round(self,env, client_num, clients, bandwidth, i):

        events = [env.process(c.train_process(env, i, client_num, clients, bandwidth))for c in clients]
        yield AnyOf(env,events)
        for c in clients:
            c.seg_transfer_time = [0]*client_num
        # else:
        #     events = [env.process(c.train_process(env, i, client_num, clients, bandwidth)) for c in clients]
        #     yield AnyOf(env, events)
        #     for c in clients:
        #         c.seg_transfer_time = [0] * client_num

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    env = simpy.Environment()
    set_up(env)
    env.run()
