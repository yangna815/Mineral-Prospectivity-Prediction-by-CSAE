# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
import argparse
from data_loader_channels import MyDataset
from network_structure import DAE  
import pandas as pd
import torch

parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=1000, type=int, help='number of epochs tp train for')
parser.add_argument('--batchsize', default=159, type=int, help='batchsize')
parser.add_argument('--sparsity_target', default=0.01, type=float, help='sparsity_target')
parser.add_argument('--Sparsity_weight', default=0.01, type=float, help='Sparsity_weight')

args = parser.parse_args()
 
class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.batchsize= config.batchsize
        self.sparsity_target= config.sparsity_target
        self.Sparsity_weight= config.Sparsity_weight

        
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        self.train_loader = MyDataset(datatxt='Data15/train/data_list.txt', transform=transforms.ToTensor())
        self.test_loader = MyDataset(datatxt='Data15/test/data_list.txt', transform=transforms.ToTensor())
        self.predict_loader = MyDataset(datatxt='Data1/predict/data_list.txt', transform=transforms.ToTensor())          
    
    def load_model(self):
        self.device = torch.device('cpu')
        self.model = DAE() 
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr) 

        
        self.criterion =  nn.MSELoss().to(self.device) 

    def train(self):
        print("train:", end='')
        self.model.train()  
        train_loss = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):  
            data= data.view(1,11,80,80)
            target=torch.tensor([target])
            self.optimizer.zero_grad() 
            # forward
            output,hidden = self.model(data)  
                        
            sparsity_target = self.sparsity_target
            p= torch.full([1,20,5,5],sparsity_target)
            q= hidden
            
            
            loss1 = self.criterion(output, data) 

            q = torch.nn.functional.softmax(q, dim=0)  
            batchsize=self.batchsize 
            q = torch.sum(q, dim=0)/batchsize 
            s1 = torch.sum(p*torch.log(p/q))
            s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
            loss2=s1+s2

            Sparsity_weight = self.Sparsity_weight
            loss_sum=loss1+Sparsity_weight*loss2
            loss= loss_sum       
  
            # backward
            loss.backward(loss.clone().detach()) 
            self.optimizer.step() 

            train_loss += loss.item()  
            total += target.size(0) #
        Loss = train_loss / len(self.train_loader)
        print('Loss: %.10f ' % Loss)
        return Loss


    
    def run(self):
        self.load_data() 
        self.load_model() #

        loss_list_train = []
        for epoch in range(1, self.epochs + 1):  

            print("\n===> epoch: %d/%d" % (epoch, self.epochs))

            train_result = self.train()  
            loss_list_train.append(train_result)
            
            if (epoch%100== 0):
                 a=str(epoch)    
                 model_out_path = "model_" +a+ ".pth"
                 torch.save(self.model, model_out_path)
                 print("Checkpoint saved to {}".format(model_out_path))            
            np.savetxt(r"结果\losstrain.txt", loss_list_train, delimiter = "\t")
        print('迭代完成')
    
    def test_with_select_model(self,model_name):
        print("test:", end='')
        model = torch.load(model_name) 
        model.eval() 
        self.load_data()
        
        test_loss=[]
       
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                
                data= data.view(1,11,80,80)
                target=torch.tensor([target])
                output,hidden = model(data)
                loss_func = nn.MSELoss()
                loss = loss_func(output, data)
                loss=loss.item()
                test_loss.append(loss) 
                
                      
        np.savetxt(r"结果\test损失.txt", test_loss, delimiter = "\t") 

    def train_with_select_model(self,model_name):
        print("train:", end='')
        model = torch.load(model_name) 
        model.eval() 
        self.load_data()
        
        train_loss=[]
       
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.train_loader):
                data= data.view(1,11,80,80)
                target=torch.tensor([target])
                output,hidden = model(data)
      
                loss_func = nn.MSELoss()

                loss = loss_func(output, data)
                loss=loss.item()
                train_loss.append(loss) 
                
             
        np.savetxt(r"结果\train损失.txt", train_loss, delimiter = "\t") 

    def predict_with_select_model(self,model_name):
        print("predict:", end='')
        model = torch.load(model_name) 
        model.eval() 
        self.load_data()
        
        predict_loss=[]
       
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.predict_loader):
                data= data.view(1,11,80,80)
                target=torch.tensor([target])
                output,hidden = model(data) 

                loss_func = nn.MSELoss()

                loss = loss_func(output, data)
                loss=loss.item()
                
                predict_loss.append(loss) 
                       
        np.savetxt(r"结果\predict损失.txt", predict_loss, delimiter = "\t")            



    
if __name__ == '__main__':
    solver = Solver(args)
    solver.run() 
    solver.test_with_select_model("model_1000.pth")  
    solver.train_with_select_model("model_1000.pth")
    solver.predict_with_select_model("model_1000.pth")