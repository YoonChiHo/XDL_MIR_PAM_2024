import random 
import time # time measurement
import datetime # date representation
import sys # write function
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision.utils import make_grid
from PIL import Image


def tensor2image(tensor, type):
    if type == 'HE': # for HE image
        image = ((tensor[0].permute(1,2,0).cpu().detach().float().numpy() * 0.5) + 0.5)*255 # numpy data [value = 0~255]
        return image.astype(np.uint8)
    else: # for PA image
        image = ((tensor[0].permute(1,2,0).cpu().detach().float().numpy() * (-0.5)) + 0.5)*255
        return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}

    def log(self, losses=None):
        self.mean_period = (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        batches_done = (self.epoch-1) * self.batches_epoch + self.batch
        batches_left = self.n_epochs * self.batches_epoch - batches_done
        time_left = datetime.timedelta(seconds = batches_left*self.mean_period)
        
        sys.stdout.write('\rEpoch %03d/%03d Batch [%04d/%04d] ETA [%s]-- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch, time_left))
        
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = np.zeros([self.n_epochs, self.batches_epoch])
                self.losses[loss_name][self.epoch-1,self.batch-1] = losses[loss_name].item()
            else:
                self.losses[loss_name][self.epoch-1,self.batch-1] = losses[loss_name].item()

            sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name][self.epoch-1,self.batch-1]))
        
        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            
            sys.stdout.write('\n')
            print('\rEpoch %03d/%03d Batch [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
            for i, loss_name in enumerate(losses.keys()):
                print('%s: %.4f -- ' % (loss_name, np.mean(self.losses[loss_name], axis=1)[self.epoch-1]))
            
            self.epoch += 1
            self.batch = 1
            
        else:
            self.batch += 1
        
        return self.losses
 
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
