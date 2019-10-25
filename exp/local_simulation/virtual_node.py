import keras
from utils import data_util
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K
Flag_aug = True
import time

class Worker:
    def __init__(self,model_json,train_data,test_data,batch_size,ori_rec_len):


        # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0)
        self.model = None

        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model =  keras.models.model_from_json(model_json)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # self.model_weights = self.model.get_weights()
        # np.save("test_model",self.model_weights)
        # exit()

        self.x_train = np.array(train_data[0])
        self.y_train = np.array(train_data[1])
        self.data_size = len(self.x_train)
        self.batch_size = batch_size
        self.datagen = data_util.data_aug(self.x_train).flow(self.x_train, self.y_train,
                                          batch_size=self.batch_size)
        self.x_test = np.array(test_data[0])
        self.y_test = np.array(test_data[1])
        # self.batch_size = int(float(batch_size * len(self.x_train))/ori_rec_len)


        print self.batch_size

    # def update_lr(self):
    #     S = K.get_value(self.model.optimizer.lr)
    #     K.set_value(self.model.optimizer.lr,S*0.1)

    def set_model_weights(self,model_weights):
        # self.update_lr()
        self.model.set_weights(model_weights)

    def train_a_step(self,step,epochs = 1):

        start_time = time.time()

        for i in range(40):
            x,y = self.datagen.next()
            self.model.train_on_batch(x,y)
        # print "time:%s"%(time.time() - start_time)
        return np.array(self.model.get_weights())



    def evaluation(self):
        # self.model.set_weights(self.model_weights)
        loss,acc = self.model.evaluate(self.x_test,self.y_test,verbose=0)
        return loss,acc

