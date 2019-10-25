import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import keras
import os
from utils import data_util
from virtual_node import Worker

class Simulator:
    def __init__(self,args):

        #data init
        [x_train,x_test] = np.load(args.dataX_path)
        [y_train,y_test] = np.load(args.dataY_path)
        ori_rec_len = len(x_train)
        parse_result = data_util.parse_distribution_info(x_train,y_train,args.data_distribution_file)
        self.max_step = args.max_step
        self.num_nodes = parse_result[0]
        if args.interval == 1:
            self.intervals = parse_result[2]
        else:
            self.intervals = np.array([1]*self.num_nodes)

        dist_file = args.data_distribution_file.split("/")[-1].split(".")[0]

        if args.interval == 1:
            self.result_root = "results/Collaborative_Epoch_Interval/%s/"%dist_file
        else:
            self.result_root = "results/Collaborative_Epoch/%s/"%dist_file
        if not os.path.exists(self.result_root):
            os.mkdir(self.result_root)
        train_data = parse_result[1]
        test_data = [x_test,y_test]

        #model init
        model_json = open(args.model_path,"r").read()

        self.avg_factor_list = [1.0 / self.num_nodes] * self.num_nodes
        self.node_intervals = [1]*self.num_nodes


        self.worker_list = []
        for i in range(self.num_nodes):
            self.worker_list.append(Worker(model_json,train_data[i],test_data,args.batch_size,ori_rec_len))

        # print self.worker_list[0].model == self.worker_list[1].model

        print("init done")

    def run(self):

        loss_result = []
        acc_result = []
        print(self.intervals)
        min_node = np.argmin(self.intervals)

        for step in range(self.max_step):
            send_flag = (step % self.intervals) == 0

            print 123

            # local update
            for worker in self.worker_list:
                worker.train_a_step(step)

            # communication

            local_updates_list = [np.array(w.get_temp_weights()) for w in self.worker_list]

            partial_aggregate_list = []

            for node in range(self.num_nodes):
                model_sum = None
                avg_sum = None
                for i in range(self.num_nodes):
                    if i == node:
                        if type(model_sum) == type(None):
                            model_sum = local_updates_list[i] * self.avg_factor_list[i]
                            avg_sum = self.avg_factor_list[i]
                        else:
                            model_sum += local_updates_list[i] * self.avg_factor_list[i]
                            avg_sum += self.avg_factor_list[i]
                        continue

                    if send_flag[i]:
                        if type(model_sum) == type(None):
                            model_sum = local_updates_list[i] * self.avg_factor_list[i]
                            avg_sum = self.avg_factor_list[i]
                        else:
                            model_sum += local_updates_list[i] * self.avg_factor_list[i]
                            avg_sum += self.avg_factor_list[i]

                partial_aggregate_list.append(model_sum / avg_sum)

            for node in range(self.num_nodes):
                self.worker_list[node].set_model_weights(partial_aggregate_list[node])



            if step % 1 == 0:
                loss,acc = self.worker_list[min_node].evaluation()
                print("step %s, loss: %s, acc:%s " % (step,loss,acc))
                loss_result.append(loss)
                acc_result.append(acc)

            if step % 5 == 0:
                np.save("%s/loss.npy"%self.result_root,loss_result)
                np.save("%s/acc.npy"%self.result_root,acc_result)


