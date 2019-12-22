import warnings
import heapq
import simpy
import argparse
from utils.args import parse_args
import simpy
import numpy as np
import random
from simpy.events import AnyOf, AllOf
import sys
from datetime import  datetime

latest_n = 5 #取最近五个预测带宽取平均
init_pridict_bandwidth = 50 #初始化预测带宽的大小
e = 0.5
CAPACITY = 100
SEG_SIZE = 30
TRAINING_TIME = 30
g = 0.01


class Client:
    
    def __init__(self, e, env,idx, clients_num, client_id, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, model=None):
        self._model = model
        self.e = e
        self.id = client_id # integer
        self.idx = idx
        self.clients_num = clients_num
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.pridict_bandwidth = []  #初始化该client到其他所有节点的预测带宽
        for i in range(self.clients_num):
            self.pridict_bandwidth.append([init_pridict_bandwidth])
        self.updates = []
        self.model1 = model.get_params()
        self.train_time = []
        self.transfer_time = []

        self.exit_bw = simpy.Container(env, init=CAPACITY, capacity=CAPACITY)
        self.record_time = [env.now]
        # self.training_time = [env.now]
        # self.seg_transfer_time = [0] * clients_num
        self.send_que = simpy.Container(env, init=0, capacity=1000)
        self.sigal = False
        # self.max_seg_transfer_time = 0
        self.round_signal = False
        self.training_time = 0

        for i in range(clients_num):
            a = []
            for j in range(clients_num):
                a.append(-1)
            self.transfer_time.append(a)
        self.args = parse_args()

    def train(self, server, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        start_time = datetime.now()
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)
        num_train_samples = len(data['y'])
        self.updates.append((num_train_samples, update))
        server.updates.append((num_train_samples, update))
        end_time = datetime.now()
        self.training_time = (end_time-start_time).seconds
        return comp, num_train_samples, update

    def update_model(self, replica, segment, server):
        print('-----update[%s]------'%self.idx)
        weight_list = []
        for p in range(segment):
            target = self.choose_best_segment(e, replica)
            segment_weight = self.get_segments(server.updates[self.idx][1], p, segment)
            print('segment:',p)
            for k in range(replica):
                segment_weight += self.get_segments(server.updates[target[k]][1], p, segment)
            print('replica done')
            segment_weight = np.array(segment_weight)
            segment_weight = segment_weight/(replica + 1)
            weight_list.extend(segment_weight)
        print('segment done')
        weight_list = np.array(weight_list)
        weight_list = self.reconstruct(weight_list)
        self.model1 = weight_list
        self.current_updates = self.updates
        self.updates = []

    def train_time_simulate(self,env, my_round, client_simulate_list, bandwidth,replica,seg):
        if my_round != 0:
            while not self.round_signal:
                yield env.timeout(0.01)
        self.round_signal = False
        print('-------------client [%d] round [%s] begin at %f:------------' % (self.idx, my_round, env.now))
        yield env.timeout(self.training_time)
        self.sigal = True
        # self.training_time.append(env.now)
        idx_list = self.get_idx_list(e, replica, seg)
        print('【Time:', env.now, '】', self.idx, 'pull from', idx_list)
        events = [env.process(self.get_transfer_time(env, client_simulate_list, bandwidth, my_round, i, seg)) for i in
                  idx_list]
        yield AllOf(env, events)
        print('【Time:', env.now, '】【Round: %s】【Id: %d】' % (my_round, self.idx), 'transfer')
        self.record_time.append(env.now)
        self.round_signal = True

    def choose_best_segment(self, e, replica):
        num = np.random.rand()
        target = []
        if self.args.algorithm == 'BACombo':
            if num < self.e:
                print('random select')
                # # client_num_list = list(range(client_num))
                # # client_num_list.remove(self.idx)
                # # return np.random.choice(client_num_list, size=k, replace=False, p=None)
                client_candidate = list(range(self.clients_num))#.remove(self.idx)
                client_candidate.remove(self.idx)
                # for i in range(replica):
                target = np.random.choice(client_candidate,size = replica,replace=False )
                    # a = np.random.randint(0, self.clients_num)
                    # while a == self.idx or a in target:
                    #     a = np.random.randint(0, self.clients_num)
                    # print('find one for ', i)
                # target.append(a)
            else:
                print('find max bandwidth')
                pridict = []
                for bandwidth in self.pridict_bandwidth:
                    if len(bandwidth) < latest_n:
                        pridict.append(np.mean(bandwidth))
                    else:
                        pridict.append(np.mean(bandwidth[-latest_n:]))
                target = list(map(pridict.index, heapq.nlargest(latest_n, pridict)))
        else:
            print('select target')
            client_candidate = list(range(self.clients_num))
            client_candidate.remove(self.idx)
            target = np.random.choice(client_candidate, size=replica, replace=False)

            # for i in range(replica):
            #     a = np.random.randint(0, self.clients_num)
            #     while a == self.idx or a in target:
            #         a = np.random.randint(0, self.clients_num)
            #     target.append(a)
        return target

    def get_segments(self, model_weights, seg, segment):
        flat_m = []
        self.shape_list = []
        print('the len of weights', len(model_weights))
        for x in model_weights:
            self.shape_list.append(x.shape)
            flat_m.extend(list(x.flatten()))
        seg_length = len(flat_m) // segment + 1

        return flat_m[seg*seg_length:(seg+1)*seg_length]

    def reconstruct(self, flat_m):
        result = []
        current_pos = 0
        print('222', self.shape_list)
        for shape in self.shape_list:
            total_number = 1
            for i in shape:
                total_number *= i
            result.append(np.array(flat_m[current_pos:current_pos+total_number]).reshape(shape))
            current_pos += total_number
        return np.array(result)

    def update_bandwidth(self,seg):
        seg_size = sys.getsizeof(self.current_updates)
        print(6666,seg_size)
        time = self.transfer_time[-1]
        for i in range(self.clients_num):
            if time[i] != -1:
                self.pridict_bandwidth[i].append((seg_size / seg)/time[i])

    def get_transfer_time(self, env, client_simulate_list, bandwidth, my_round, idx_list,seg):
        events = [env.process(self.pull_seg(env, client_simulate_list[i], i, bandwidth, my_round, seg))
                  for i in idx_list]
        yield AllOf(env, events)
        # self.max_seg_transfer_time = np.max(self.seg_transfer_time)

    def get_idx_list(self, e, replica, seg):
        idx_list = []
        for i in range(seg):
            target = self.choose_best_segment(e, replica)
            idx_list.append(target)
        return idx_list
        # client_num_list = list(range(client_num))
        # client_num_list.remove(self.idx)
        # return np.random.choice(client_num_list, size=k, replace=False, p=None)

    def pull_seg(self, env, client_simulate, idx, bandwidth, my_round, seg):
        while not client_simulate.sigal:
            yield env.timeout(0.01)
        yield client_simulate.send_que.put(1)
        exit_bw = client_simulate.exit_bw.level
        bottleneck_num = client_simulate.send_que.level
        link_bw = bandwidth[idx][self.idx]
        start_time = env.now
        yield env.process(self.que_monitor(env, client_simulate, exit_bw, link_bw, bottleneck_num, seg))
        end_time = env.now
        self.transfer_time[self.idx][client_simulate.idx] = end_time-start_time
        yield client_simulate.send_que.get(1)
        print('【Time:', env.now,
              '】【Round: %s】【Id: %d】-----pulling-----【Id：%d】' % (my_round, self.idx, client_simulate.idx))

    def que_monitor(self, env, client_simulate, exit_bw, link_bw, bottleneck_num, seg):
        seg_size = sys.getsizeof(self.current_updates)
        residual_seg_size = seg_size / seg
        final_bw = np.min([exit_bw / bottleneck_num, link_bw])
        last_que = bottleneck_num
        while residual_seg_size >= 0:
            yield env.timeout(g)
            residual_seg_size -= g * final_bw
            current_que = client_simulate.send_que.level
            if current_que != last_que:
                last_que = current_que
                final_bw = np.min([exit_bw / current_que, link_bw])
            else:
                last_que = current_que

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test':
            data = self.eval_data
        metrics = self.model.test(data)
        metrics['time'] = self.record_time[-1]
        return metrics

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
