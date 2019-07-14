import os
import matplotlib.pyplot as plt
from datetime import datetime

from keras.callbacks import Callback

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        current = logs.get(self.monitor)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % batch)
            self.model.stop_training = True

class TimeClock(Callback):

    def __init__(self):
        super().__init__()
        self.start = False
        self.i = 0

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        if not self.start:
            print("Start to clock.")
            self.start = True
            self.start_time = datetime.now()
            self.i = 0
    def on_epoch_end(self, epoch, logs=None):
        self.end_time = datetime.now()
        self.i+=1

        c = self.end_time-self.start_time
        hour = c.total_seconds() // 3600 * 3600
        minute = (c.total_seconds() - hour) // 60 * 60
        second = c.total_seconds() - hour - minute

        print(f"In epoch {self.i}, took {int(hour//3600)} hours, {int(minute//60)} minutes, {int(second)} seconds.")

class Lossplot(Callback):
    def __init__(self,model_name,save_dir = "./loss_image"):
        super().__init__()
        self.model_name = model_name
        self.save_dir = save_dir
        self.reset()
        self.start = False
        self.i = 0
        self.epoch = 0
        self.ilis = []

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        if not self.start:
            self.start = True
            print("Start to record loss value.")

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.losses['batch'].append(logs.get('loss'))
        self.i+=1

    def on_epoch_end(self, batch, logs=None):
        self.losses['epoch'].append(logs.get('loss'))
        self.ilis.append(self.i)
        self.epoch+=1
        self.loss_plot()


    def loss_plot(self):
        plt.figure()
        plt.plot(self.losses['epoch'])
        for i in self.ilis:
            plt.plot([i-1],[self.losses['batch'][i-1]],"r+")
        plt.grid(True)
        plt.xlabel(self.model_name)
        plt.ylabel('loss_epoch')

        os.makedirs(self.save_dir,exist_ok=True)
        save_path = os.path.join(self.save_dir,f"{self.model_name}_epoch_{self.epoch}_step_{len(self.losses['batch'])}.png")

        plt.savefig(save_path)

        print(f"loss curve ploted in {save_path}.")

    def reset(self):
        self.losses = {'batch': [], 'epoch': []}

class LossReportor(Callback):
    '''用于汇报每次loss的提升幅度'''
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.start = False
        self.min_loss = None


    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        if not self.start:
            self.start = True
            print("Start to record loss value.")

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, batch, logs=None):
        if self.min_loss is None:
            self.min_loss = logs.get('loss')
            print(f"loss from nan to {self.min_loss}")
        else:
            if self.min_loss < logs.get('loss'):
                print(f"loss does't improved.")
            else:
                print(f"loss imporved from {self.min_loss} to {logs.get('loss')}")
                self.min_loss = logs.get('loss')





