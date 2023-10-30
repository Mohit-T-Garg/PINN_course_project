import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
class PhysicsInformedNN:

    def __init__(self, x, y , t , u , v , layers):
        X = torch.cat([x,y,t],axis=1)
        self.device = "cuda:0"
        self.X = X.to(self.device)
        
        self.x = X[:,0:1].to(self.device)
        self.y = X[:,1:2].to(self.device)
        self.t = X[:,2:3].to(self.device)
        
        self.u = u.to(self.device)
        self.v = v.to(self.device)
        
        self.layers = layers
        self.model =  torch.nn.Sequential(
                torch.nn.Linear(3, 20),
                torch.nn.Tanh(),
                torch.nn.Linear(20, 20),
                torch.nn.Tanh(),
                torch.nn.Linear(20, 20),
                torch.nn.Tanh(),
                torch.nn.Linear(20, 20),
                torch.nn.Tanh(),
                torch.nn.Linear(20, 2))       
        self.lambda_1 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True).to(self.device)
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.01),requires_grad=True).to(self.device)

        self.total_loss = []#total loss history
        self.u_pred= []
        self.v_pred= []
        self.l1= []
        self.l2 = []
        self.f_u_pred= []
        self.f_v_pred = [] 
        self.optimizer = None
        

    def costFunction(self, x, y , t): 
        """Compute the cost function."""

        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        loss_u = torch.mean((self.u - u_pred)**2)
        loss_v = torch.mean((self.v - v_pred)**2)
        loss_f_u = torch.mean((f_u_pred)**2)
        loss_f_v = torch.mean((f_v_pred)**2)
        
        return loss_u , loss_v , loss_f_u , loss_f_v
    
    
    def net_NS(self, x, y, t):
    psi_and_p = self.model(torch.cat([x,y,t],axis=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v = -torch.autograd.grad(psi, x , grad_outputs=torch.ones_like(psi),create_graph=True, retain_graph=True, allow_unused=True)[0][0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True, allow_unused=True)[0]
    nu = 0.01/np.pi
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True, allow_unused=True)[0]
    f_u = u_t + u*u_x + v*u_y - nu*(u_xx + u_yy) + p_x
    f_v = v_t + u*v_x + v*v_y - nu*(v_xx + v_yy) + p_y
    return f_u, f_v


    def closure(self):
        
        self.optimizer.zero_grad()#set gradients to 0
        loss_u , loss_v , loss_f_u , loss_f_v= self.costFunction(self.x, self.y , self.t)#calculate loss
        total_loss  = loss_u + loss_v + 1*loss_f_u + 1*loss_f_v
        total_loss.backward(retain_graph=True)# backpropagate for derivative of loss
        return total_loss
    
    def predict(self, x, y, t):
    psi_and_p = self.model(torch.cat([x, y, t], dim=1))
    psi = psi_and_p[:, 0:1]
    p = psi_and_p[:, 1:2]

    u = torch.autograd.grad(
        psi,
        y,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    v = -torch.autograd.grad(
        psi,
        x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    return u.to(self.device), v.to(self.device), p.to(self.device)
    
    def train(self, epochs):
        """Train the model."""

        '''
        This function is used for training the network. While updating the model params use "costFunction" 
        function for calculating loss
        Params:
            epochs - number of epochs

        This function doesn't have any return values. Just print the losses during training
        
        '''
        self.optimizer = torch.optim.LBFGS(chain(self.model.parameters(),[self.lambda_1,self.lambda_2]), lr = 0.08)

        for r in range(0,epochs):
            
            loss_u , loss_v , loss_f_u , loss_f_v= self.costFunction(self.x.to(self.device), self.y.to(self.device) , self.t.to(self.device)) #calculate loss
            loss_u , loss_v , loss_f_u , loss_f_v = loss_u.item() , loss_v.item() , loss_f_u.item() , loss_f_v.item()
            total_loss  = loss_u + loss_v + 1*loss_f_u + 1*loss_f_v
            self.u_pred.append(loss_u)
            self.v_pred.append(loss_v)
            self.f_u_pred.append(loss_f_u)
            self.f_v_pred.append(loss_f_v)
            self.total_loss.append(total_loss)
            self.l1.append(self.lambda_1.detach().numpy())
            self.l2.append(self.lambda_2.detach().numpy())
            print("Epoch: ",r,"/",epochs-1," Total loss = ",self.total_loss[-1],"Loss @ {u,v,f_u,f_v}",self.u_pred[-1],self.v_pred[-1],self.f_u_pred[-1],self.f_v_pred[-1])
            self.optimizer.step(self.closure)#closure here

    