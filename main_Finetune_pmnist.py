import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sn 
import pandas as pd 
import random
# import pbd 
import argparse, time
import math 
from copy import deepcopy

class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = nn.Linear(784,n_hidden,bias=False)
        self.lin2 = nn.Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = nn.Linear(n_hidden, n_outputs, bias=False)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        self.act['Lin1']=x
        x = self.lin1(x)        
        x = F.relu(x)
        self.act['Lin2']=x
        x = self.lin2(x)        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        self.act['fc1']=x
        x = self.fc1(x)
        return x 
    
def get_model(model):
    return deepcopy(model.state_dict())
    
def set_model(model, state_dict):
    model.load_state_dict(deepcopy(state_dict)) 
    return
fisher_dict = {}
optpar_dict = {}
def train(args, model, device, task_id, x, y, optimizer, criterion):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output, target)  
        loss.backward()
        optimizer.step()


def test (args, model, device, x, y, criterion):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def main(args):
    tstart = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from dataloader import pmnist as pmd 
    data, taskcla, inputsize = pmd.get(seed=args.seed, pc_valid=args.pc_valid)
    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id=0
    task_list = []
    for k, nclas in taskcla:
        threshold = np.array([0.95, 0.99, 0.99])

        print('-'*100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('-'*100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        model = MLPNet(args.n_hidden, args.n_outputs).to(device)
        print ('Model parameters ---')
        for k_t, (m, param) in enumerate(model.named_parameters()):
            print (k_t,m,param.shape)
        print ('-'*40)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        for epoch in range(1, args.n_epochs+1):
            # Train 
            clock0=time.time()
            train(args, model,device,task_id, xtrain, ytrain,optimizer,criterion)
            clock1=time.time()
            tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                    tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()
        
        # Test 
        test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
        
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion) 
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        # update task id 
        task_id +=1
    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    # Plots
    
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with BASELINE')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    # parser.add_argument('--ewc_lambda',default=0.35,type=float,
    #                     help='ewc_lambda')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')

    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)
                





