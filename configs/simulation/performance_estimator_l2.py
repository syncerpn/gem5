import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim

import os

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import tqdm

import argparse
parser = argparse.ArgumentParser(description='Supercom: performance estimator')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch_step', type=int, default=50, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=150, help='total epochs to run')

parser.add_argument('--split_rate', type=int, default=5, help='total epochs to run')
parser.add_argument('--dataset_file', default='/home/nghiant/git/gem5/dataset_2x4_all.txt', help='dataset')
#output
parser.add_argument('--cv_dir', default='pe_model/test_6', help='checkpoint directory (models and logs are saved here)')

args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print('# setting learning_rate to %.2E'%lr)

class performancer_estimator(nn.Module):
    def __init__(self, len_x, len_y=1):
        super(performancer_estimator, self).__init__()
        self.fcs = nn.ModuleList()

        self.fcs.append(nn.Linear(len_x, 128))
        self.fcs.append(nn.Linear(128, 128))
        self.fcs.append(nn.Linear(128, 128))
        self.fcs.append(nn.Linear(128, len_y))
        self.init_network()

    def init_network(self):
        self.fcs[0].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fcs[0].weight)
        self.fcs[1].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fcs[1].weight)
        self.fcs[2].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fcs[2].weight)
        self.fcs[3].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fcs[3].weight)

    def forward(self, x):
        z = F.relu(self.fcs[0](x))
        z = F.relu(self.fcs[1](z))
        z = F.relu(self.fcs[2](z))
        z = self.fcs[3](z)
        return z

def train(epoch):
    #turn on train mode on agent
    core.train()

    total_loss = 0

    #walk through the train set
    for batch_idx in tqdm.tqdm(range(n_batch)):
        #get input batch and push to GPU
        xt = torch.Tensor(Xt[shuffle_index[batch_idx,:], :])
        yt = torch.Tensor(Yt[shuffle_index[batch_idx,:], :])
        
        xt = Variable(xt).cuda()
        yt = Variable(yt).cuda()

        #forward input through agent
        y = core(xt)

        loss = (y - yt)**2
        loss = loss.sum() / xt.shape[0]
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #performance stats
    log_str = '[ INFO ] E: %d | LOSS: %.6f' % (epoch, total_loss)
    print(log_str)
    
def valid(epoch):
    #turn on eval mode on agent
    core.eval()

    #walk through the test set
    xv = Variable(torch.Tensor(Xv)).cuda()
    yv = Variable(torch.Tensor(Yv)).cuda()
    y = core(xv)

    # diff = torch.abs(y - yv)
    diff = torch.abs(y - yv) / yv * 100

    diff_mean = diff.mean()
    diff_std = diff.std()

    # log_str = '[ INFO ] VALID - DIFF_MEAN: %.6f | DIFF_STD: %.6f' % (diff_mean, diff_std)
    log_str = '[ INFO ] VALID - DIFF_MEAN: %.3f | DIFF_STD: %.3f' % (diff_mean, diff_std)
    print(log_str)

    state = {
        'core': core.state_dict(),
        'epoch': epoch,
        'diff': diff_mean,
    }
    torch.save(state, args.cv_dir + '/ckpt_E_%d_D_%.6f.t7' % (epoch, diff_mean))

def test():
    #turn on eval mode on agent
    core.eval()

    #walk through the test set
    xv = Variable(torch.Tensor(Xv)).cuda()
    yv = Variable(torch.Tensor(Yv)).cuda()
    y = core(xv)

    with open('pe_order_check.txt', 'w') as f:
        y_list = y.flatten().tolist()
        f.write(' '.join([str(i) for i in y_list]))
        f.write('\n')
        yv_list = yv.flatten().tolist()
        f.write(' '.join([str(i) for i in yv_list]))

    with open('pe_order_check_sort.txt', 'w') as f:
        sy_list = torch.argsort(y.flatten()).tolist()
        f.write(' '.join([str(i) for i in sy_list]))
        f.write('\n')
        syv_list = torch.argsort(yv.flatten()).tolist()
        f.write(' '.join([str(i) for i in syv_list]))

### main
print('[ INFO ] load dataset from %s' % (args.dataset_file))
with open(args.dataset_file, 'r') as f:
    dataset = np.loadtxt(f, delimiter=' ')

len_x = dataset.shape[1] - 1
n_datapoint = dataset.shape[0]

trainset = dataset[0:n_datapoint*args.split_rate//10, :]
Xt = trainset[:, 0:len_x]
Yt = trainset[:, len_x:len_x+1]

validset = dataset[n_datapoint*args.split_rate//10: , :]
Xv = validset[:, 0:len_x]
Yv = validset[:, len_x:len_x+1]

n_sample = Xt.shape[0]

## model
# create and load core model
core = performancer_estimator(len_x)

n_batch = n_sample // args.batch_size
n_sample = n_batch * args.batch_size
print('[ INFO ] trainset contains %d samples ~ %d batches x %d samples per batch' % (n_sample, n_batch, args.batch_size))

# cuda config
core.cuda()

# optimizer and lr schedule
optimizer = optim.Adam(core.parameters(), lr=args.lr)
lr_scheduler = LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)

# shuffle trainset
shuffle_index = np.arange(n_sample).reshape([n_batch, -1])

for epoch in range(0, args.max_epochs + 1):
    np.random.shuffle(shuffle_index)
    lr_scheduler.adjust_learning_rate(epoch)

    if epoch % 1 == 0:
        valid(epoch)

    train(epoch)

test()