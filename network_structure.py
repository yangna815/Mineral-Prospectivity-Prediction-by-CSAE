# -*- coding: utf-8 -*-

import torch.nn as nn
class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encoder0=nn.Conv2d(11,10, kernel_size=3, stride=2,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.encoder2=nn.Conv2d(10,20, kernel_size=3, stride=1,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.encoder4=nn.Conv2d(20, 40, kernel_size=3,stride=1,padding=1)
        self.encoder5=nn.Conv2d(40, 40, kernel_size=3,stride=1,padding=1)
        self.encoder6=nn.Conv2d(40, 20, kernel_size=3, stride=1,padding=1)
        self.pool7=nn.MaxPool2d(kernel_size=2, return_indices=True)
        
        self.unpool7=nn.MaxUnpool2d((2,2),2)
        self.decoder6= nn.ConvTranspose2d(20,40,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.decoder5= nn.ConvTranspose2d(40,40,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.decoder4= nn.ConvTranspose2d(40,20,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.unpool3=nn.MaxUnpool2d((2,2),2)
        self.decoder2= nn.ConvTranspose2d(20,10,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.unpool1=nn.MaxUnpool2d((2,2),2)
        self.decoder0= nn.ConvTranspose2d(10,11,kernel_size=3,stride=2,padding=1,output_padding=1)
        
        
    def forward(self, x):
        encoder0 = self.encoder0(x)
        encoder0_size = encoder0.size() 
        pool1, indices1 = self.pool1(encoder0)
         
        encoder2 = self.encoder2(pool1)
        encoder2_size = encoder2.size()
        pool3, indices3 = self.pool3(encoder2)
        
        encoder4 = self.encoder4(pool3)
        encoder5 = self.encoder5(encoder4)
        encoder6 = self.encoder6(encoder5)
        encoder6_size = encoder6.size() 
        pool7, indices7 = self.pool7(encoder6)
        
        unpool7 = self.unpool7(input=pool7, indices=indices7, output_size=encoder6_size)
        decoder6 = self.decoder6(input=unpool7)
        decoder5 = self.decoder5(input=decoder6)
        decoder4 = self.decoder4(input=decoder5)

        unpool3 = self.unpool3(input=decoder4, indices=indices3, output_size=encoder2_size)
        
        decoder2 = self.decoder2(input=unpool3)
        unpool1 = self.unpool1(input=decoder2, indices=indices1, output_size=encoder0_size)
        decoder0 = self.decoder0(input=unpool1) 
                      
        return decoder0, pool7
