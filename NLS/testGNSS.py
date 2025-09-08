import os
import warnings
import torch
import numpy as np
from pyDOE import lhs
from collections import OrderedDict
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmath  # 用于复数运算

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建文件夹函数
def make_folder(folder_name):
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
make_folder('data')
make_folder('log')
make_folder('model')
make_folder('png')

# --------------------------- 数据生成 --------------------------- #
def solution(x, t):
    """KdV方程解析解"""
    term1 = cmath.exp(12*t)
    term2 = -1j * cmath.exp(4j*x + 4*(1+1j)*t)
    term3 = cmath.exp(4*(1+1j)*x + 4*(2+1j)*t)
    term4 = -1j * cmath.exp(4*x)
    N = term1 + term2 + term3 + term4
    D = np.cos(4*(t+x)) - np.cosh(4*(x-t)) - 2*np.cosh(8*t)
    coefficient = 2*(1-1j)
    exponential = cmath.exp(-2*(1+1j)*(x+3*t))
    psi_2 = coefficient * (N/D) * exponential
    return psi_2.real, psi_2.imag, abs(psi_2)

def generate_data():
    x = np.linspace(-5, 5, 512)
    t = np.linspace(-1, 1, 201)
    X, T = np.meshgrid(x, t)
    Exact_u = np.zeros_like(X)
    Exact_v = np.zeros_like(X)
    Exact_h = np.zeros_like(X)
    for i in range(len(t)):
        for j in range(len(x)):
            r,i_part,ab = solution(x[j], t[i])
            Exact_u[i,j] = r
            Exact_v[i,j] = i_part
            Exact_h[i,j] = ab
    return x[:,None], t[:,None], Exact_u, Exact_v, Exact_h, X, T

# --------------------------- 网络模块 --------------------------- #
class Scale(torch.nn.Module):
    def __init__(self, lb, ub):
        super(Scale, self).__init__()
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
    def forward(self, X):
        return 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers)-1
        self.activation = torch.nn.Tanh
        layer_list = []
        for i in range(self.depth-1):
            layer_list.append(('layer_%d'%i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d'%i, self.activation()))
        layer_list.append(('layer_%d'%(self.depth-1), torch.nn.Linear(layers[-2], layers[-1])))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))
    def forward(self, x):
        return self.layers(x)

def xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class FileWriter:
    def __init__(self, logfilename):
        self.file_path = os.path.join(current_dir,'log',logfilename)
    def write(self, content, mode='a'):
        with open(self.file_path, mode, encoding='utf-8') as f:
            f.write(content+'\n')

# --------------------------- PINN --------------------------- #
class PhysicsInformedNN():
    def __init__(self, layers, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.dnn = torch.nn.Sequential(Scale(lb, ub), DNN(layers)).to(device)
        self.dnn.apply(xavier)
    
    def set_training_data(self, x0, u0, v0, tb, Xf):
        self.x0 = torch.tensor(x0).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        self.v0 = torch.tensor(v0).float().to(device)
        self.tb = torch.tensor(tb).float().to(device)
        self.xf = torch.tensor(Xf[:,0:1], requires_grad=True).float().to(device)
        self.tf = torch.tensor(Xf[:,1:2], requires_grad=True).float().to(device)

    # 这里改成 PyTorch 操作
        self.x_lb = torch.full_like(self.tb, self.lb[0])
        self.x_ub = torch.full_like(self.tb, self.ub[0])
        self.t0 = torch.full_like(self.x0, -1)


    def forward(self, X):
        return self.dnn(X)

    def net_f(self, x, t):
        output_uv = self.forward(torch.concat([x,t], dim=1))
        u = output_uv[:,0:1]
        v = output_uv[:,1:2]
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx,x,grad_outputs=torch.ones_like(u_xx),retain_graph=True,create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx,x,grad_outputs=torch.ones_like(u_xxx),retain_graph=True,create_graph=True)[0]
        u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]

        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_xxx = torch.autograd.grad(v_xx,x,grad_outputs=torch.ones_like(v_xx),retain_graph=True,create_graph=True)[0]
        v_xxxx = torch.autograd.grad(v_xxx,x,grad_outputs=torch.ones_like(v_xxx),retain_graph=True,create_graph=True)[0]
        v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]

        fu = u_t + 0.5*v_xx + (u**2 + v**2)*v +0.0625*(v_xxxx + 6*v*(v_x**2-u_x**2)+12*u*u_x*v_x +4*v*(u_x**2+v_x**2)+6*(u**2+v**2)*v_xx+6*(u**2+v**2)**2*v) -0.125*(u_xxx+6*(u**2+v**2)*u_x)
        fv = v_t - 0.5*u_xx - (u**2 + v**2)*u -0.0625*(u_xxxx +6*u*(u_x**2-v_x**2)+12*v*u_x*v_x+4*u*(u_x**2+v_x**2)+10*(u**2+v**2)*u_xx+6*(u**2+v**2)**2*u) -0.125*(v_xxx+6*(u**2+v**2)*v_x)
        return fu, fv

    def backward(self):
        # 初始条件
        output0 = self.forward(torch.concat([self.x0,self.t0],dim=1))
        u0_pred = output0[:,0:1]
        v0_pred = output0[:,1:2]
        loss0 = torch.mean((u0_pred-self.u0)**2) + torch.mean((v0_pred-self.v0)**2)
        # 边界条件
        output_lb = self.forward(torch.concat([self.x_lb,self.tb],dim=1))
        output_ub = self.forward(torch.concat([self.x_ub,self.tb],dim=1))
        u_lb_pred = output_lb[:,0:1]
        v_lb_pred = output_lb[:,1:2]
        u_ub_pred = output_ub[:,0:1]
        v_ub_pred = output_ub[:,1:2]
        loss_b = torch.mean((u_lb_pred-u_ub_pred)**2) + torch.mean((v_lb_pred-v_ub_pred)**2)
        # PDE 残差
        f_u, f_v = self.net_f(self.xf,self.tf)
        loss_f = torch.mean(f_u**2) + torch.mean(f_v**2)
        # 总损失
        loss = loss0 + loss_b + loss_f
        loss.backward()
        if self.iter % 100 == 0:
            msg = f"Iter: {self.iter}, Loss: {loss.item():.5e}, Loss0: {loss0.item():.5e}, Loss_b: {loss_b.item():.5e}, Loss_f: {loss_f.item():.5e}"
            if hasattr(self,'writer'):
                self.writer.write(msg)
            print(msg)
        self.iter += 1
        return loss

    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000):
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
            tolerance_change=1.0*np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.logfilename = logfilename
        if logfilename is not None:
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
        print('Training time: %.4f' % (time.time()-start_time))

    def save_model(self, filename):
        save_path = os.path.join(current_dir,'model',filename)
        torch.save(self.dnn.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}.")

    def load_model(self, filename):
        load_path = os.path.join(current_dir,'model',filename)
        self.dnn.load_state_dict(torch.load(load_path))
        print(f"Model parameters loaded from {load_path}.")

    def predict(self, X_star):
        X_star = torch.tensor(X_star).float().to(device)
        self.dnn.eval()
        op_star = self.dnn(X_star)
        u_star = op_star[:,0:1].detach().cpu().numpy()
        v_star = op_star[:,1:2].detach().cpu().numpy()
        return u_star, v_star

# --------------------------- 主函数 --------------------------- #
if __name__ == "__main__":
    # 生成训练数据
    x, t, Exact_u, Exact_v, Exact_h, X, T = generate_data()
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None]
    v_star = Exact_v.flatten()[:,None]
    h_star = Exact_h.flatten()[:,None]

    lb = X_star.min(0)
    ub = X_star.max(0)
    N0, Nb, Nf = 50, 50, 20000
    layers = [2,50,50,50,50,2]

    x0 = x
    u0 = Exact_u[0,:].T[:,None]
    v0 = Exact_v[0,:].T[:,None]
    idx = np.random.choice(x0.shape[0], N0, replace=False)
    x0 = x0[idx,:]; u0 = u0[idx,:]; v0 = v0[idx,:]
    idt = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[idt,:]
    Xf = lb + (ub-lb)*lhs(2,Nf)

    # 定义模型
    model = PhysicsInformedNN(layers, lb, ub)
    model.set_training_data(x0, u0, v0, tb, Xf)
    model.train(5000, logfilename='log-2sol.txt')
    model.save_model('PINN-2sol.pth')

    # 预测
    model.load_model('PINN-2sol.pth')
    u_pred, v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print(f'Error u: {error_u:e}, Error v: {error_v:e}, Error h: {error_h:e}')

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact_h - H_pred)

    # --------------------------- 绘图 --------------------------- #
    X0 = np.concatenate([x0, 0*x0], axis=1)
    X_lb = np.concatenate([0*tb+lb[0], tb], axis=1)
    X_ub = np.concatenate([0*tb+ub[0], tb], axis=1)
    X_boundary = np.vstack([X0, X_lb, X_ub])

    fig = plt.gcf()
    gs0 = gridspec.GridSpec(1,2)
    gs0.update(top=1-0.06,bottom=1-1/3,left=0.1,right=0.9,wspace=0.3)
    ax0 = plt.subplot(gs0[0,0])
    h0 = ax0.imshow(H_pred.T, interpolation='nearest', cmap='rainbow', extent=[t.min(),t.max(),x.min(),x.max()], origin='lower', aspect='auto')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right",size="5%",pad=0.05)
    fig.colorbar(h0,cax=cax0)
    ax1 = plt.subplot(gs0[0,1])
    h1 = ax1.imshow(Error.T, interpolation='nearest', cmap='rainbow', extent=[t.min(),t.max(),x.min(),x.max()], origin='lower', aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right",size="5%",pad=0.05)
    fig.colorbar(h1,cax=cax1)
    plt.show()
