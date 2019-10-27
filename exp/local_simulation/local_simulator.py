import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import keras
import os
from utils import data_util
from .virtual_node  import Worker

class Simulator:
    def __init__(self,args):

        #data init
        [x_train,x_test] = np.load(args.dataX_path,encoding="latin1")
        [y_train,y_test] = np.load(args.dataY_path,encoding="latin1")
        ori_rec_len = len(x_train)
        parse_result = data_util.parse_distribution_info(x_train,y_train,args.data_distribution_file)
        self.max_step = args.max_step
        self.num_nodes = parse_result[0]
        self.step_epoch = args.step_epoch
        self.segments = args.seg
        self.replica = args.rep
        self.p2p = args.p2p
        self.shape_list = None
        if args.p2p:
            if args.seg == 1:
                self.result_file = "%s_gossip_rep_%s.npy"%(self.num_nodes,self.replica)
            else:
                self.result_file = "%s_seg_%s_rep_%s.npy"%(self.num_nodes,self.segments,self.replica)
        else:
            print ("PS mode")
            self.result_file = "%s_baseline.npy"%self.num_nodes
        self.intervals = np.array([1]*self.num_nodes)

        dist_file = args.data_distribution_file.split("/")[-1].split(".")[0]

        train_data = parse_result[1]
        test_data = [x_test,y_test]

        #model init
        model_json = open(args.model_path,"r").read()


        self.worker_list = []
        for i in range(self.num_nodes):
            self.worker_list.append(Worker(model_json,train_data[i],test_data,args.batch_size,ori_rec_len))

        current_model = self.worker_list[0].model.get_weights()
        for g in self.worker_list:
            g.set_model_weights(current_model)
            # g.model.summary()
            # exit()

        # print self.worker_list[0].model == self.worker_list[1].model

        print("init done")


    def get_segments(self,model_weights,seg):
        flat_m = []
        self.shape_list = []
        for x in model_weights:
            self.shape_list.append(x.shape)
            flat_m.extend(list(x.flatten()))

        seg_length = len(flat_m) // self.segments + 1

        return flat_m[seg*seg_length:(seg+1)*seg_length]

    def reconstruct(self,flat_m):
        result = []
        current_pos = 0
        for shape in self.shape_list:
            total_number = 1
            for i in shape:
                total_number *= i
            result.append(np.array(flat_m[current_pos:current_pos+total_number]).reshape(shape))
            current_pos += total_number
        return np.array(result)

    def early_step(self):
        pass



    def run(self):

        acc_result = []
        print(self.intervals)

        for step in range(self.max_step):
            partial_aggregate_list = []
            # local update
            for w_idx,worker in enumerate(self.worker_list):
                current_model = worker.train_a_step(0,self.step_epoch)
                partial_aggregate_list.append(current_model)

            # global aggregation


            if self.p2p:
                parts = self.segments
                replica = self.replica

                for node in range(self.num_nodes):
                    self_model = partial_aggregate_list[node]
                    model_len = len(self_model)
                    part_len = int(np.ceil(float(model_len) / parts))
                    trive_model = []
                    visited_list = []
                    for r in range(replica):
                        weight_list = []
                        for p in range(parts):
                            target = np.random.randint(0,self.num_nodes)

                            while target == node or (target,p) in visited_list:
                                target = np.random.randint(0,self.num_nodes)
                            visited_list.append((target,p))
                            weight_list.extend(self.get_segments(partial_aggregate_list[target],p))
                        trive_model.append(self.reconstruct(weight_list))

                    model_sum = np.array(self_model)
                    for m in trive_model:
                        model_sum += np.array(m)

                    self.worker_list[node].set_model_weights(model_sum / (replica+1))

            else:

                model_sum = None
                for weights in partial_aggregate_list:
                    if type(model_sum) == type(None):
                        model_sum = np.array(weights)
                    else:
                        model_sum += weights

                final_model = model_sum / len(partial_aggregate_list)

                for node in range(self.num_nodes):
                    self.worker_list[node].set_model_weights(final_model)


            if step % 1 == 0:
                accs = []
                for node in range(self.num_nodes):
                    loss,acc = self.worker_list[node].evaluation()
                    accs.append(acc)
                print("step %s, acc:%s " % (step,np.mean(accs)))
                acc_result.append(accs)

            if step % 10 == 0:
                # np.save("%s/loss.npy"%self.result_root,loss_result)
                np.save(self.result_file,acc_result)
        np.save(self.result_file,acc_result)

