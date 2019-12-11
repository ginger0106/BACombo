import random
import warnings
import numpy as np
import heapq
import argparse
from utils.args import parse_args

latest_n = 5 #取最近五个预测带宽取平均
init_pridict_bandwidth = 50 #初始化预测带宽的大小
e = 0.5


class Client:
    
    def __init__(self, idx, clients_num, client_id, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, model=None):
        self._model = model
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
        for i in range(clients_num):
            a = []
            for j in range(clients_num):
                a.append(0.02)
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
        return comp, num_train_samples, update

    def update_model(self, replica, segment, server):
        trive_model = []
        weight_list = []
        for p in range(segment):
            target = self.choose_best_segment(e, replica)
            segment_weight = self.get_segments(server.updates[self.idx][1], p)
            for k in range(replica):
                segment_weight += self.get_segments(server.updates[target[k]][1], p)
            segment_weight = np.array(segment_weight)
            segment_weight = segment_weight/(replica + 1)
            weight_list.extend(segment_weight)
        weight_list = np.array(weight_list)
        trive_model.append(self.reconstruct(weight_list))

        self.model1 = np.array(trive_model)

        self.updates = []
        

    def choose_best_segment(self, e, replica):
        num = np.random.rand()
        target = []
        if self.args.algorithm == 'BACombo':
            if num < e:
                for i in range(replica):
                    a = np.random.randint(0, self.clients_num)
                    while a == self.idx or a in target:
                        a = np.random.randint(0, self.clients_num)
                    target.append(a)
            else:
                pridict = []
                for bandwidth in self.pridict_bandwidth:
                    if len(bandwidth) < latest_n:
                        pridict.append(np.mean(bandwidth))
                    else:
                        pridict.append(np.mean(bandwidth[-latest_n:]))
                target = list(map(pridict.index, heapq.nlargest(latest_n, pridict)))
        else:
            for i in range(replica):
                a = np.random.randint(0, self.clients_num)
                while a == self.idx or a in target:
                    a = np.random.randint(0, self.clients_num)
                target.append(a)
        return target

    def get_segments(self, model_weights, seg):
        flat_m = []
        self.shape_list = []
        for x in model_weights:
            self.shape_list.append(x.shape)
            flat_m.extend(list(x.flatten()))
        seg_length = len(flat_m) // self.args.segment + 1

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

    def update_bandwidth(self):
        time = self.transfer_time[-1]
        for i in range(self.clients_num):
            if time[i] != -1:
                self.pridict_bandwidth[i].append(1/time[i])

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
        return self.model.test(data)

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
