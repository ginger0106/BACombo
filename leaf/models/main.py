"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


from utils.args import parse_args
from utils.model_utils import read_data
from simpy.events import AnyOf, AllOf
import simpy



STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

class set_up():
    def __init__(self, env):
        self.args = parse_args()
        self.env = env

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + self.args.seed)
        np.random.seed(12 + self.args.seed)
        tf.set_random_seed(123 + self.args.seed)

        self.model_path = '%s/%s.py' % (self.args.dataset, self.args.model)
        if not os.path.exists(self.model_path):
            print('Please specify a valid dataset and a valid model.')
        self.model_path = '%s.%s' % (self.args.dataset, self.args.model)

        print('############################## %s ##############################' % self.model_path)
        self.mod = importlib.import_module(self.model_path)
        self.ClientModel = getattr(self.mod, 'ClientModel')

        self.tup = MAIN_PARAMS[self.args.dataset][self.args.t]
        self.num_rounds = self.args.num_rounds if self.args.num_rounds != -1 else self.tup[0]
        self.eval_every = self.args.eval_every if self.args.eval_every != -1 else self.tup[1]

        # Suppress tf warnings
        tf.logging.set_verbosity(tf.logging.WARN)

        # Create 2 models
        self.model_params = MODEL_PARAMS[self.model_path]
        if self.args.lr != -1:
            self.model_params_list = list(self.model_params)
            self.model_params_list[0] = self.args.lr
            self.model_params = tuple(self.model_params_list)

        # Create client model, and share params with server model
        tf.reset_default_graph()
        self.client_model = self.ClientModel(self.args.seed, *self.model_params)

        # Create clients
        self.clients = setup_clients(self.args.aggregation, self.args.e,self.env,self.args.dataset, self.client_model)
        # Create server
        self.server = Server(self.client_model, len(self.clients))
        self.client_ids, self.client_groups, self.client_num_samples = self.server.get_clients_info(self.clients)
        print('Clients in Total: %d' % len(self.clients))
        
        self.replica = self.args.replica
        self.segment = self.args.segment
        self.client_num = len(self.clients)

        self.main_proc = env.process(self.main_process())


    def round_proc(self, my_round):
        random_num = np.random.rand()
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in self.clients}
        for c in self.clients:
            print('-----training-[%s]-------'%c.idx)
            comp, num_samples, update = c.train(self.server, num_epochs=self.args.num_epochs, batch_size=self.args.batch_size,
                                                minibatch=self.args.minibatch)
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
        self.sys_writer_fn(my_round + 1,self.c_ids, sys_metrics, self.c_groups, self.c_num_samples)
        # for c in self.clients:
        #     print('start straining client:',c.idx)
        #     c.train(self.server, num_epochs=self.args.num_epochs, batch_size=self.args.batch_size,
        #             minibatch=self.args.minibatch)
        if self.args.algorithm == 'gossip':
            print(3333)
            for c in self.clients:
                c.update_model(self.args.replica, 1, self.server,random_num,my_round)
                # c.metrics = c.test(my_round,'test')
        elif self.args.algorithm == 'combo':
            for c in self.clients:
                c.update_model(self.args.replica, self.args.segment, self.server,random_num,my_round,self.args.adam_lr)
                # c.metrics = c.test(my_round,'test')
        elif self.args.algorithm == 'BACombo':

            for c in self.clients:
                c.update_model(self.args.replica, self.args.segment, self.server,random_num,my_round)

            for c in self.clients:
                c.update_bandwidth(self.args.segment)

        self.server.updates = []
        # print_metrics(test_stat_metrics, num_samples, prefix='test_')

        # self.server.model = self.clients[0].model1
        print(4444)
        # self.stat_writer_fn = get_stat_writer_function(self.client_ids, self.client_groups, self.client_num_samples,
        #                                                self.args)
        events = [self.env.process(c.train_time_simulate(self.env,  my_round, self.clients, self.server.bandwidth,
                                                         self.replica,self.args.segment,
                                                         random_num)) for c in self.clients]
        print(555)
        yield AllOf(self.env,events)


    def main_process(self):
        if self.args.algorithm == 'fedavg':
            clients_per_round = self.args.clients_per_round if self.args.clients_per_round != -1 else self.tup[2]
            # Initial status
            print('--- Random Initialization ---1111')
            self.stat_writer_fn = get_stat_writer_function(self.client_ids, self.client_groups, self.client_num_samples, self.args)
            self.sys_writer_fn = get_sys_writer_function(self.args)
            print_stats(self.env,0, self.server, self.clients, self.client_num_samples, self.args, self.stat_writer_fn)
        else:
            print('--- Random Initialization ---2222')
            self.stat_writer_fn = get_stat_writer_function(self.client_ids, self.client_groups, self.client_num_samples, self.args)
            self.sys_writer_fn = get_sys_writer_function(self.args)
            # print_stats(0, self.server, self.clients, self.client_num_samples, self.args, self.stat_writer_fn)
            clients_per_round = len(self.clients)

        for i in range( self.num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1,  self.num_rounds, clients_per_round))
            # Select clients to train this round
            self.server.select_clients(i, online(self.clients), num_clients=clients_per_round)
            self.c_ids, self.c_groups, self.c_num_samples = self.server.get_clients_info(self.server.selected_clients)
            print(111)
            if self.args.algorithm == 'fedavg':
                # Simulate server model training on selected clients' data
                self.sys_metrics = self.server.train_model(num_epochs=self.args.num_epochs, batch_size=self.args.batch_size, minibatch=self.args.minibatch)
                self.sys_writer_fn(i + 1, self.c_ids, self.sys_metrics, self.c_groups, self.c_num_samples)
                # Update server model
                self.server.update_model()
            else:
                print(2222)
                yield self.env.process(self.round_proc(i))
                print(10000000)
                # for c in self.clients:
                #     c.train(self.server, num_epochs=self.args.num_epochs, batch_size=self.args.batch_size, minibatch=self.args.minibatch)
                # if self.args.algorithm == 'gossip':
                #     for c in self.clients:
                #         c.update_model(self.args.replica, 1, self.server)
                # elif self.args.algorithm == 'combo':
                #     for c in self.clients:
                #         c.update_model(self.args.replica, self.args.segment, self.server)
                # elif self.args.algorithm == 'BACombo':
                #     for c in self.clients:
                #         c.update_model(self.args.replica, self.args.segment, self.server)
                #     for c in self.clients:
                #         c.update_bandwidth()
                # self.server.updates = []
                # self.server.model = self.clients[0].model
    
            # Test model
            signal = [c.test_signal for c in self.clients]  #
            if  (i + 1) == self.num_rounds or (i + 1) % self.eval_every == 0:
                self.stat_writer_fn = get_stat_writer_function(self.client_ids, self.client_groups, self.client_num_samples,
                                                          self.args)

                print_stats(self.env,i+1, self.server, self.clients, self.client_num_samples, self.args, self.stat_writer_fn)

        # Save server model
        ckpt_path = os.path.join('checkpoints', self.args.dataset)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        save_path = self.server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(self.args.model)))
        print('Model saved in path: %s' % save_path)
    
        # Close models
        self.server.close_model()

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(aggregation,e,env,users, groups, train_data, test_data, model):
    args = parse_args()
    if len(groups) == 0:
        groups = [[] for _ in users]
    a = [i for i in range(len(users))]
    if args.algorithm == 'fedavg':
        clients = [Client(aggregation,e,env,j, len(users), u, g, train_data[u], test_data[u], model) for j, u, g in zip(a, users, groups)]
    else:
        clients = [Client(aggregation,e,env, j, len(users), u, g, train_data[u], test_data[u], model) for j, u, g in
                   zip(a, users, groups)]
        # clients = [Client(env,j, args.clients_per_round, u, g, train_data[u], test_data[u], model) for j, u, g in zip(a, users, groups)]
        # clients = clients[:args.clients_per_round]
    return clients


def setup_clients(aggregation,e,env,dataset, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    print(9898334353252435,len(users))

    clients = create_clients(aggregation,e,env,users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(env,num_round, server, clients, num_samples, args, writer):
    args = parse_args()
    if args.algorithm == 'fedavg':
        train_stat_metrics = server.test_model(clients, set_to_use='train')
        print_metrics(train_stat_metrics, num_samples, prefix='train_')
        writer(num_round, train_stat_metrics, 'train')
        test_stat_metrics = server.test_model(clients, set_to_use='test')
        print_metrics(test_stat_metrics, num_samples, prefix='test_')
        writer(num_round, test_stat_metrics, 'test')
    else:
        train_stat_metrics = {}
        for client in clients:
            #client.model1 = server.model
            c_metrics = client.test(num_round,'train')
            train_stat_metrics[client.id] = c_metrics
        print_metrics(train_stat_metrics, num_samples, prefix='train_')
        writer(num_round, train_stat_metrics, 'train')
        test_stat_metrics = {}
        for client in clients:
            #client.model1 = server.model
            c_metrics = client.test(num_round,'test')
            test_stat_metrics[client.id] = c_metrics
        print_metrics(test_stat_metrics, num_samples, prefix='test_')
        writer(num_round, test_stat_metrics, 'test')

# def save_stats():


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that cient.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    env = simpy.Environment()
    set_up(env)
    env.run()
    # main()
