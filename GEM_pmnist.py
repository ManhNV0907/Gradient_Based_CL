import torch
# from torch._C import R 
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

import quadprog

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

    def get_params(self):
        self.vars = []
        for p in list(self.parameters()):
            if p.requires_grad:
                self.vars.append(p)
        return self.vars
    
def get_model(model):
    return deepcopy(model.state_dict())
    
def set_model(model, state_dict):
    model.load_state_dict(deepcopy(state_dict)) 
    return

# def update_buffer(x, y, x_mem, y_mem):
#     r=np.arange(x.size(0))
#     np.random.shuffle(r)
#     r=torch.LongTensor(r).cuda()
#     b =r[:200] 
#     x_mem_new = x[b].view(-1,28*28)
#     y_mem_new = y[b]
#     x_mem = torch.cat((x_mem, x_mem_new))
#     y_mem = torch.cat((y_mem, y_mem_new))
#     return x_mem, y_mem
def update_buffer(x, y, x_mem, y_mem, task_id):
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).cuda()
    b =r[:200] 
    x_mem_new = {}
    y_mem_new = {}
    x_mem_new[task_id] = x[b].view(-1,28*28).cuda()
    y_mem_new[task_id] = y[b].cuda()
    x_mem[task_id] = torch.cat((x_mem[task_id].cuda(), x_mem_new[task_id]))
    y_mem[task_id] = torch.cat((y_mem[task_id].cuda(), y_mem_new[task_id]))
    return x_mem, y_mem
# def update_gradient_matrix(g, task_id, model):
#     g[task_id-1] = []
#     for param in model.parameters():
#         g[task_id-1].append( param.grad.clone())
#     return g
def grad_to_vector(model):
    vec = []
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n,p in params.items():
        if p.grad is not None:
            vec.append(p.grad.view(-1))
        else:
            # Part of the network might has no grad, fill zero for those terms
            vec.append(p.data.clone().fill_(0).view(-1))
    return torch.cat(vec)
def vector_to_grad(model, vec):
    # Overwrite current param.grad by slicing the values in vec (flatten grad)
    pointer = 0
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in params.items():
        # The length of the parameter
        num_param = p.numel()
        if p.grad is not None:
            # Slice the vector, reshape it, and replace the old data of the grad
            p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
            # Part of the network might has no grad, ignore those terms
        # Increment the pointer
        pointer += num_param

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    P = P + G * 0.001
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    new_grad = torch.Tensor(x).view(-1)
    new_grad = new_grad.cuda()
    return new_grad

def train (args, model, device, x, y, optimizer,criterion, x_mem, y_mem, task_id):
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
    x_mem, y_mem = update_buffer(x, y, x_mem, y_mem, task_id)
def train_gem(args, model, device, task_id, x, y, optimizer, criterion, x_mem, y_mem, task_grads):
    
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        params = [p for p in model.parameters() if p.requires_grad]
        # gradient computed using current batch
        current_grad_vec = grad_to_vector(model)
        for i in range(0, len(x_mem)):
          optimizer.zero_grad()
          x_mem[i], y_mem[i] = x_mem[i].view(-1,28*28).to(device), y_mem[i].to(device)
          mem_logit = model(x_mem[i])
          # y_mem = torch.LongTensor(y_mem)
          loss_mem = criterion(mem_logit, y_mem[i].long())
          loss_mem.backward()
          #gradient of previous task
          task_grads[i] = grad_to_vector(model)
        #check violation
        mem_grad_vec = torch.stack(list(task_grads.values()))
        dotp = current_grad_vec * mem_grad_vec
        dotp = dotp.sum(dim=1)
        if (dotp < 0).sum() != 0:
            new_grad = project2cone2(current_grad_vec, mem_grad_vec)
            vector_to_grad(model, new_grad)






        # gradient computed using memory samples
        # grad_ref = [p.grad.clone() for p in params]
        #update gradient matrix
        # for p in params:
        #     gradient_matrix[task_id-1].append(p)
        # update_gradient_matrix(gradient_matrix, task_id, model)


        # indx = torch.cuda.LongTensor(task_list[:-1])
        # prod = torch.mm(grad[:, task_id].unsqueeze(0),
        #                     gradient_matrix.index_select(1, indx))
        
        # inner product of grad and grad_ref
        # for grad_ref in gradient_matrix.value():
        #     prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
        # if (prod < 0).sum() !=0:
        #     project2cone2(grad[:, task_id].unsqueeze(1), gradient_matrix)
            # prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
            # # do projection
            # grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
        # replace params' grad
        # for g, p in zip(grad, params):
        #     p.grad.data.copy_(g)
        optimizer.step()
    #update mem
    x_mem, y_mem = update_buffer(x, y, x_mem, y_mem, task_id)

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
    fisher_dict = {}
    task_grads = {}
    optpar_dict = {}
    memory = {}
    task_mem_cache = {}
    memory_size = 300
    x_mem = {}
    y_mem = {}
    # for k, v in obj.items():
    #   res[k] = move_to(v, device)
    for i in range(10):
      x_mem[i] = torch.Tensor([])
      y_mem[i] = torch.Tensor([])
    observed_tasks= []
    
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
    n_memories = 200
    for k, nclas in taskcla:
        # gradient_matrix = {}
        # task_grads = {}
        
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
        if task_id==0:
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            print ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            print ('-'*40)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, x_mem, y_mem, task_id)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                print()
            # Test
            print ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_gem(args, model,device,task_id, xtrain, ytrain,optimizer,criterion, x_mem, y_mem, task_grads)
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
    parser = argparse.ArgumentParser(description='Sequential PMNIST with EWC')
    parser.add_argument('--batch_size_train', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--ewc_lambda',default=0.35,type=float,
                        help='ewc_lambda')
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
                





