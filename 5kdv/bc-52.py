import os
current_dir = os.path.dirname(os.path.abspath(__file__))
def make_folder(folder_name):
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
make_folder('data')     # 创建文件夹用于保存文件
make_folder('log')
make_folder('model')
make_folder('png')
import warnings
warnings.filterwarnings('ignore')
import torch
if torch.cuda.is_available():   # 设置设备名称
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
import scipy.io
import numpy as np
np.random.seed(1234)
from pyDOE import lhs
from collections import OrderedDict
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In one space dimension, the Burger’s equation along with Dirichlet boundary conditions reads as
#     u_t + u u_x - (0.01/pi) u_xx = 0,       x in [-1, 1], t in [0, 1]

#     u(0,x)=-sin(pi x)
#     u(t,-1)=u(t,1)=0
# Let us define f(t, x) to be given by
#     f := u_t + u u_x - (0.01/pi) u_xx,
# and proceed by approximating u(t, x) by a deep neural network.


class Scale(torch.nn.Module):
    def __init__(self, lb, ub):
        super(Scale, self).__init__()
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
    def forward(self, X):
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        return H

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
    def forward(self, x):
        out = self.layers(x)
        return out

class FileWriter:
    def __init__(self, logfilename):
        self.file_path = os.path.join(current_dir, 'log', logfilename)
    def write(self, content, mode='a'):
        with open(self.file_path, mode, encoding='utf-8') as file:
            file.write(content + '\n')

def xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class PhysicsInformedNN():
    def __init__(self, layers, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.dnn = torch.nn.Sequential(Scale(lb, ub), DNN(layers)).to(device)
        self.dnn.apply(xavier)
    
    def set_training_data(self, x0, u0, tb, Xf):
        # 对于不同的问题，修改这里
        self.x0 = torch.tensor(x0).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        
        self.tb = torch.tensor(tb).float().to(device)
        self.xf = torch.tensor(Xf[:, 0:1], requires_grad=True).float().to(device)
        self.tf = torch.tensor(Xf[:, 1:2], requires_grad=True).float().to(device)
        

        self.x_lb = torch.tensor(0*tb + lb[0]).float().to(device)
        self.x_ub = torch.tensor(0*tb + ub[0]).float().to(device)
        # self.t0 = torch.tensor(0*x0).float().to(device)
        self.t0 = torch.full_like(self.x0, -0.5).to(device)

    
    def backward(self):
        # 对于不同的问题，修改这里

        
        u0_pred = self.dnn(torch.concat([self.x0, self.t0], dim=1))  # shape: (N, 2)
        
        loss0 = torch.mean(torch.square(u0_pred - self.u0))
                


        u_lb_pred = self.dnn(torch.concat([self.x_lb, self.tb], dim=1))
        u_ub_pred = self.dnn(torch.concat([self.x_ub, self.tb], dim=1))
        
       
        
        
        loss_b = torch.mean(torch.square(u_lb_pred - u_ub_pred)) 
                 

        f_u = self.net_f(self.xf, self.tf)
        loss_f = torch.mean(torch.square(f_u)) 

        loss = loss0 + loss_b + loss_f
        loss.backward()

        if self.iter % 100 == 0:
            # 记录信息的格式可能也要改
            message = 'Iter: %d, Loss: %.5e, Loss_0: %.5e, Loss_b: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss0.item(), loss_b.item(), loss_f.item())
            if self.logfilename is not None:
                self.writer.write(message)
            print(message)
        self.iter += 1
        return loss

    def net_f(self, x, t):  # 在backward中调用
        u = self.dnn(torch.concat([x, t], dim=1))
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), retain_graph=True, create_graph=True)[0]
        u_xxxxx = torch.autograd.grad(u_xxxx, x, grad_outputs=torch.ones_like(u_xxxx), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        return u_t + 180*u*u*u_x + 30*u_x*u_xx + 30*u*u_xxx + u_xxxxx


    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000): # 固定格式，不用修改
        self.dnn.train()
        self.iter = init_iter
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=LBFGS_max_iter, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.logfilename = logfilename
        if self.logfilename is not None:
            self.writer = FileWriter(logfilename)
            self.writer.write("Start Training:", mode='w')
        start_time = time.time()
        for _ in range(nIter):
            self.optimizer_Adam.zero_grad()
            self.backward()
            self.optimizer_Adam.step()
        def LBFGS_callback():
            self.optimizer_LBFGS.zero_grad()
            return self.backward()
        self.optimizer_LBFGS.step(LBFGS_callback)
        elapsed = time.time() - start_time
        if self.logfilename is not None:
            self.writer.write('Training time: %.4f' % (elapsed))
        print('Training time: %.4f' % (elapsed))

    def load_model(self, filename): # 固定格式，不用修改
        if filename is not None:
            load_path = os.path.join(current_dir, 'model', filename)
            self.dnn.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}.")

    def save_model(self, filename): # 固定格式，不用修改
        if filename is not None:
            save_path = os.path.join(current_dir, 'model', filename)
            torch.save(self.dnn.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}.")      

    def predict(self, X_star): # 一般不改
        X_star = torch.tensor(X_star).float().to(device)
        self.dnn.eval()
        u_star = self.dnn(X_star)
        
        u_star = u_star.detach().cpu().numpy()
       
        return u_star

class bcPhysicsInformedNN():
    def __init__(self, layers, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.dnn = torch.nn.Sequential(Scale(lb, ub), DNN(layers)).to(device)
        self.dnn.apply(xavier)
    
    def set_training_data(self, x0, u0, t0,  u1, tb, Xf, Xh):
        # 对于不同的问题，修改这里
        self.x0 = torch.tensor(x0).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        
        self.u1 = torch.tensor(u1).float().to(device)
        
        self.tb = torch.tensor(tb).float().to(device)
        self.xf = torch.tensor(Xf[:, 0:1], requires_grad=True).float().to(device)
        self.tf = torch.tensor(Xf[:, 1:2], requires_grad=True).float().to(device)
        self.x1 = torch.tensor(Xh[:, 0:1], requires_grad=True).float().to(device)
        self.t1 = torch.tensor(Xh[:, 1:2], requires_grad=True).float().to(device)
        

        self.x_lb = torch.tensor(0*tb + lb[0]).float().to(device)
        self.x_ub = torch.tensor(0*tb + ub[0]).float().to(device)
        self.t0 = torch.full_like(self.x0, t0).to(device)
    

    
    def backward(self):
        # 对于不同的问题，修改这里

        
        u0_pred = self.dnn(torch.concat([self.x0, self.t0], dim=1))  # shape: (N, 2)
    
       
        loss0 = torch.mean(torch.square(u0_pred - self.u0)) 
                

        u_lb_pred = self.dnn(torch.concat([self.x_lb, self.tb], dim=1))
        u_ub_pred = self.dnn(torch.concat([self.x_ub, self.tb], dim=1))
        
        
        
        
        loss_b = torch.mean(torch.square(u_lb_pred - u_ub_pred))



        u1_pred = self.dnn(torch.concat([self.x1, self.t1], dim=1))

        loss1 = torch.mean(torch.square(u1_pred - self.u1))

        f_u = self.net_f(self.xf, self.tf)
        loss_f = torch.mean(torch.square(f_u))

        loss = loss0 + loss_b + loss_f + loss1
        loss.backward()

        if self.iter % 100 == 0:
            # 记录信息的格式可能也要改
            message = 'Iter: %d, Loss: %.5e, Loss_0: %.5e, Loss_1: %.5e, Loss_b: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss0.item(), loss1.item(),  loss_b.item(), loss_f.item())
            if self.logfilename is not None:
                self.writer.write(message)
            print(message)
        self.iter += 1
        return loss

    def net_f(self, x, t):  # 在backward中调用
        u = self.dnn(torch.concat([x, t], dim=1))
        
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), retain_graph=True, create_graph=True)[0]
        u_xxxxx = torch.autograd.grad(u_xxxx, x, grad_outputs=torch.ones_like(u_xxxx), retain_graph=True, create_graph=True)[0]
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

       

        fu = u_t + 180*u*u*u_x + 30*u_x*u_xx + 30*u*u_xxx + u_xxxxx

        return fu


    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000): # 固定格式，不用修改
        self.dnn.train()
        self.iter = init_iter
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=LBFGS_max_iter, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.logfilename = logfilename
        if self.logfilename is not None:
            self.writer = FileWriter(logfilename)
            self.writer.write("Start Training:", mode='w')
        start_time = time.time()
        for _ in range(nIter):
            self.optimizer_Adam.zero_grad()
            self.backward()
            self.optimizer_Adam.step()
        def LBFGS_callback():
            self.optimizer_LBFGS.zero_grad()
            return self.backward()
        self.optimizer_LBFGS.step(LBFGS_callback)
        elapsed = time.time() - start_time
        if self.logfilename is not None:
            self.writer.write('Training time: %.4f' % (elapsed))
        print('Training time: %.4f' % (elapsed))

    def load_model(self, filename): # 固定格式，不用修改
        if filename is not None:
            load_path = os.path.join(current_dir, 'model', filename)
            self.dnn.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}.")

    def save_model(self, filename): # 固定格式，不用修改
        if filename is not None:
            save_path = os.path.join(current_dir, 'model', filename)
            torch.save(self.dnn.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}.")      

    def predict(self, X_star): # 一般不改
        X_star = torch.tensor(X_star).float().to(device)

        self.dnn.eval()
        op_star = self.dnn(X_star)
        u_star = op_star[:, 0:1]
        u_star = u_star.detach().cpu().numpy()
        return u_star


if __name__ == "__main__":
    # 对于不同的问题，只需要更改准备训练数据部分，以及set_training_data，backward 函数
    # 如果是多输出的问题，可能还要改 predict 函数（对于burgers，它只有一维的输出）

    #############################################
    ############## 准备训练数据 ##################
    #############################################


    data = scipy.io.loadmat(os.path.join(current_dir, 'data', '52[-0.5,1][-10,15].mat'))
    t = data['t'].T.flatten()[:,None]     # (T, 1) = (201,1)
    print(t.shape)
    x = data['x'].T.flatten()[:,None]     # (X, 1) = (512,1)
    print(x.shape)
    Exact_u = np.real(data['uu'])   # (T, X)
    print(Exact_u.shape)
    


    X, T = np.meshgrid(x, t)            # (T, X)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))  # (T*X, 2)
    u_star = Exact_u.flatten()[:,None]                                # (T*X, 1)
    
    
    lb = X_star.min(0)          # [-1, 0]       lb: lower boundary
    ub = X_star.max(0)          # [1, 0.99]     ub: upper boundary

    N0 = 50         #x取点
    Nf = 10000                          # 区域内的配点个数
    iterations = 10000

    layers = [2, 50,50,50,50,50, 1]

    x0 = X[0:1, :].T                           # (X, 1)
    u0 = Exact_u[0:1, :].T                       # (X, 1)
    idx = np.random.choice(x0.shape[0], N0, replace=False) 
    x0 = x0[idx, :]                             # (N0, 1)
    u0 = u0[idx, :]
     



    start  = 0
    step = 50
    stop = 201
    steps_lb = np.arange(0,stop+step,step)
    steps_ub = 1 + steps_lb

    # counter = 0

    

    # for i in range(0, steps_lb.size - 1):
    #     print(f"\n=== Training Segment {i} ===")
    #     t1 = steps_lb[i]
    #     #print(t1)
    #     t2 = steps_ub[i + 1]
    #     #print(t2)
    #     temp_t = t[:t2, :] #从初始时刻到小区间右边界上的时间网格点
    #     #print(temp_t)
    #     t_seg = t[t1:t2, :] #每个小区间上的时间网格点
    #     print(t_seg)


    # #构造空间-时间网格
    #     X, T = np.meshgrid(x, t)
    #     X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # #随机选择边界时间点
    #     Nb = int(step / 2)
    #     idt = np.random.choice(t_seg.shape[0], Nb, replace=False)
    #     tb = t_seg[idt, :] #tb shape: (25, 1)(Nb,1)是从每个小区间随机采样的25个点
    #     # print("tb shape:",tb.shape)
    #     # print(tb)

    #     if counter == 0:
    #         # Domain bounds
    #         lb = np.array([-10, -0.5])
    #         ub = np.array([15, np.max(t_seg)])
            
    #         temp_lb = np.array([-10, np.min(t_seg)])
    #         temp_ub = np.array([15, np.max(t_seg)])
            
    #         Xf = temp_lb +  (temp_ub-temp_lb)*lhs(2,Nf)
    #         # u_star = np.tile(u0,(len(t_seg),1))
    #         # v_star = np.tile(v0,(len(t_seg),1))
            
    #         logfilename = "log-bc[-0.5,1][-10,15].txt"
    #         globals()[f"model_{counter}"] = PhysicsInformedNN(layers, lb, ub)
    #         globals()[f"model_{counter}"].set_training_data(x0, u0, tb, Xf)
    #         start_time = time.time()                
    #         globals()[f"model_{counter}"].train(iterations,logfilename)
    #         elapsed = time.time() - start_time                
    #         print('Training time: %.4f' % (elapsed))
    #         xh, th = np.meshgrid(x,temp_t)
    #         print("x_old shape:",xh.shape)
    #         print("t_old shape:",th.shape)

    #         Xh = np.hstack((xh.flatten()[:,None], th.flatten()[:,None]))
    #         print("Xh shape:",Xh.shape)
    #         pred = globals()[f"model_{counter}"].predict(Xh)
    #         u1 = pred
    #     else:
    #         # Domain bounds
    #         # For any step other than initial step
    #         lb = np.array([-10, -0.5])
    #         ub = np.array([15, np.max(t_seg)])
            
    #         temp_lb = np.array([-10, np.min(t_seg)])
    #         temp_ub = np.array([15, np.max(t_seg)])
            
    #         u0 = u1[-N0:] # 从末尾开始取最后 Nx 个元素
    #         t0 = t1   # t0变成小区间左端点
    #         t0 = t0.item()  # 提取标量值
            
    #         Xf = temp_lb + (temp_ub - temp_lb)*lhs(2,Nf)
    #         # u_star = np.tile(u0,(len(t_seg),1))
    #         # v_star = np.tile(v0,(len(t_seg),1))

    #         logfilename = f"log-bc52[-0.5,1][-10,15]{i}.txt"
            
    #         globals()[f"model_{counter}"] = bcPhysicsInformedNN(layers, lb, ub)
    #         globals()[f"model_{counter}"].set_training_data(x0, u0, t0,  u1, tb, Xf, Xh)
    #         start_time = time.time()                
    #         globals()[f"model_{counter}"].train(iterations,logfilename)
    #         elapsed = time.time() - start_time                
    #         print('Training time: %.4f' % (elapsed))
    #         xh, th = np.meshgrid(x,temp_t)
    #         Xh = np.hstack((xh.flatten()[:,None], th.flatten()[:,None]))
    #         pred = globals()[f"model_{counter}"].predict(Xh)
    #         u1 = pred


    # # 保存模型
        
    #     model_name = f"52[-0.5,1][-10,15]_{i}.pth"
    #     globals()[f"model_{counter}"].save_model(model_name)

    #     counter += 1




                                



    

    # ###########################################
    # ########### 测试模型时取消注释 #############
    # ###########################################

    model = bcPhysicsInformedNN(layers, lb, ub)


    model_name = "52[-0.5,1][-10,15]_3.pth"
    model.load_model(model_name)

    u_pred = model.predict(X_star)      # (T*X, 1)
    
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)  # L2 error
    
    print("u_star.shape:", u_star.shape)
    print("u_pred.shape:", u_pred.shape)
    


    print('Error u: %e' % (error_u))
    

    X, T = np.meshgrid(x,t)



    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    
    # print("H_pred.shape:", H_pred.shape)
    
    # Error = np.abs(Exact_h - H_pred)

    # # #####################################################################
    # # ############################ Plotting ###############################
    # #####################################################################


    lb = X_star.min(0)          # [-1, 0]       lb: lower boundary
    print(lb)
    ub = X_star.max(0)          # [1, 0.99]     ub: upper boundary

    idx_x = np.random.choice(x.shape[0], 100, replace=False)
    x0 = x[idx_x, :]
    idx_t = np.random.choice(t.shape[0], 100, replace=False)
    tb = t[idx_t, :]


    X0 = np.concatenate([x0, 0*x0], axis=1)
    X_lb = np.concatenate([0*tb + lb[0], tb], axis=1)
    X_ub = np.concatenate([0*tb + ub[0], tb], axis=1)
    X_boundary = np.vstack([X0, X_lb, X_ub])

    ################## Row 0: u(t,x) ##################
    ################## Row 0: u(t,x) 和 Error ##################
    fig = plt.gcf()
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.1, right=0.9, wspace=0.3)

# ---- 左图：U_pred ----
    ax0 = plt.subplot(gs0[0, 0])
    h0 = ax0.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',  
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')##'rainbow'
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h0, cax=cax0)
    # ax0.plot(X_boundary[:,1], X_boundary[:,0], 'kx', label = 'Data (%d points)' % (X_boundary.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    # ax0.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax0.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[100]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax0.plot(t[125]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[150]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax0.plot(t[175]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$x$')
    ax0.legend(frameon=False, loc = 'best')
    ax0.set_title('$Pred\ u(t,x)$', fontsize = 10)

# ---- 右图：误差图 Error = |Exact - Pred| ----
    Error = np.abs(Exact_u - U_pred)
    ax1 = plt.subplot(gs0[0, 1])
    h1 = ax1.imshow(Error.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h1, cax=cax1)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax1.set_title('$|Exact - Pred|$', fontsize = 10)

    ################# Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_u[50,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x, U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = -0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-10.1,15.1])
    ax.set_ylim([-0.01,0.5])
    ax.set_aspect( (15 - (-10)) / (0.5 - 0), adjustable='box' )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_u[100,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-10.1,15.1])
    ax.set_ylim([-0.01,0.5])
    ax.set_aspect( (15 - (-10)) / (0.5 - 0), adjustable='box' )
    ax.set_title('$t = 0$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_u[150,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[150,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-10.1,15.1])
    ax.set_ylim([-0.01,0.5])
    ax.set_aspect( (15 - (-10)) / (0.5 - 0), adjustable='box' )
    ax.set_title('$t = 0.25$', fontsize = 10)

    fig.savefig(os.path.join(current_dir, 'png', 'bc 52[-0.5,1][-10,15].png'))
    plt.show()

