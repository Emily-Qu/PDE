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
    device = torch.device('cuda')
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
    
    def set_training_data(self, x0, u0, v0, tb, Xf):
        self.x0 = torch.tensor(x0).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        self.v0 = torch.tensor(v0).float().to(device)
        self.tb = torch.tensor(tb).float().to(device)
        self.xf = torch.tensor(Xf[:, 0:1], requires_grad=True).float().to(device)
        self.tf = torch.tensor(Xf[:, 1:2], requires_grad=True).float().to(device)
        
        self.x_lb = torch.tensor(0*tb + lb[0]).float().to(device)
        self.x_ub = torch.tensor(0*tb + ub[0]).float().to(device)
        self.t0 = torch.full_like(self.x0, 0).to(device)
    
    def backward(self):
        output0 = self.dnn(torch.concat([self.x0, self.t0], dim=1))  # shape: (N, 2)
        u0_pred = output0[:, 0:1]  # shape: (N, 1)
        v0_pred = output0[:, 1:2]  # shape: (N, 1)
        loss0 = torch.mean(torch.square(u0_pred - self.u0)) + \
                torch.mean(torch.square(v0_pred - self.v0))

        output_lb = self.dnn(torch.concat([self.x_lb, self.tb], dim=1))
        output_ub = self.dnn(torch.concat([self.x_ub, self.tb], dim=1))
        u_lb_pred = output_lb[:, 0:1]
        v_lb_pred = output_lb[:, 1:2]
        u_ub_pred = output_ub[:, 0:1]
        v_ub_pred = output_ub[:, 1:2]
        
        
        
        loss_b = torch.mean(torch.square(u_lb_pred - u_ub_pred))  + \
                 torch.mean(torch.square(v_lb_pred - v_ub_pred)) 

        f_u, f_v = self.net_f(self.xf, self.tf)
        loss_f = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))


        loss = loss0 + loss_b + loss_f
        loss.backward()

        if self.iter % 100 == 0:
            message = 'Iter: %d, Loss: %.5e, Loss_0: %.5e, Loss_b: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss0.item(), loss_b.item(), loss_f.item())
            if self.logfilename is not None:
                self.writer.write(message)
            print(message)
        self.iter += 1
        return loss

    def net_f(self, x, t):
        output_uv = self.dnn(torch.concat([x, t], dim=1))
        u = output_uv[:, 0:1]
        v = output_uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        # u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        # u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), retain_graph=True, create_graph=True)[0]
        # u_xxxxx = torch.autograd.grad(u_xxxx, x, grad_outputs=torch.ones_like(u_xxxx), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        # v_xxx = torch.autograd.grad(v_xx, x, grad_outputs=torch.ones_like(v_xx), retain_graph=True, create_graph=True)[0]
        # v_xxxx = torch.autograd.grad(v_xxx, x, grad_outputs=torch.ones_like(v_xxx), retain_graph=True, create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        fu = u_t + 0.5*v_xx + (u**2 + v**2)*v
        fv = v_t - 0.5*u_xx - (u**2 + v**2)*u  
        
        return fu, fv


    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000):
        self.dnn.train()
        self.iter = init_iter
        self.optimizer_Adam = torch.optim.AdamW(self.dnn.parameters())
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=LBFGS_max_iter, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-7, 
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

    def load_model(self, filename):
        if filename is not None:
            load_path = os.path.join(current_dir, 'model', filename)
            self.dnn.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}.")

    def save_model(self, filename):
        if filename is not None:
            save_path = os.path.join(current_dir, 'model', filename)
            torch.save(self.dnn.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}.")      

    def predict(self, X_star): # 一般不改
        X_star = torch.tensor(X_star).float().to(device)
        self.dnn.eval()
        op_star = self.dnn(X_star)
        u_star = op_star[:, 0:1]
        v_star = op_star[:, 1:2]
        u_star = u_star.detach().cpu().numpy()
        v_star = v_star.detach().cpu().numpy()
        return u_star, v_star

class IbcPhysicsInformedNN():
    def __init__(self, layers, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.dnn = torch.nn.Sequential(Scale(lb, ub), DNN(layers)).to(device)
        self.dnn.apply(xavier)
        self.earliest_solutions = []  # 存储每个子域的最早解
    
    def set_training_data(self, x0, u0,v0, tb, Xf, earliest_solutions=None):
        self.x0 = torch.tensor(x0).float().to(device)
        self.u0 = torch.tensor(u0).float().to(device)
        self.v0 = torch.tensor(v0).float().to(device)
        
        self.tb = torch.tensor(tb).float().to(device)
        self.xf = torch.tensor(Xf[:, 0:1], requires_grad=True).float().to(device)
        self.tf = torch.tensor(Xf[:, 1:2], requires_grad=True).float().to(device)
        
        self.x_lb = torch.tensor(0*tb + lb[0]).float().to(device)
        self.x_ub = torch.tensor(0*tb + ub[0]).float().to(device)
        self.t0 = torch.full_like(self.x0, t0_val).to(device)
        
        # 存储每个子域的最早解 (X, u) 元组列表
        self.earliest_solutions = earliest_solutions if earliest_solutions else []
    
    def backward(self):
        output0 = self.dnn(torch.concat([self.x0, self.t0], dim=1))  # shape: (N, 2)
        u0_pred = output0[:, 0:1]  # shape: (N, 1)
        v0_pred = output0[:, 1:2]  # shape: (N, 1)
        loss0 = torch.mean(torch.square(u0_pred - self.u0)) + \
                torch.mean(torch.square(v0_pred - self.v0))


        output_lb = self.dnn(torch.concat([self.x_lb, self.tb], dim=1))
        output_ub = self.dnn(torch.concat([self.x_ub, self.tb], dim=1))
        u_lb_pred = output_lb[:, 0:1]
        v_lb_pred = output_lb[:, 1:2]
        u_ub_pred = output_ub[:, 0:1]
        v_ub_pred = output_ub[:, 1:2]
        
        
        
        loss_b = torch.mean(torch.square(u_lb_pred - u_ub_pred))  + \
                 torch.mean(torch.square(v_lb_pred - v_ub_pred)) 

        f_u, f_v = self.net_f(self.xf, self.tf)
        loss_f = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))

        
        # Ibc-PINN核心改进1: 使用每个子域的最早解作为参考
        loss_s = 0
        if self.earliest_solutions:
            for X_ref, h_ref in self.earliest_solutions:
                h_pred_ref = self.dnn(torch.concat([X_ref[:, 0:1], X_ref[:, 1:2]], dim=1))
                loss_s += torch.mean(torch.square(h_pred_ref - h_ref))
        
        loss = loss0 + loss_b + loss_f + loss_s
        loss.backward()
        
        if self.iter % 100 == 0:
            message = 'Iter: %d, Loss: %.5e, Loss_0: %.5e, Loss_b: %.5e, Loss_f: %.5e, Loss_s: %.5e' % \
                     (self.iter, loss.item(), loss0.item(), loss_b.item(), loss_f.item(), loss_s.item() if self.earliest_solutions else 0)
            if self.logfilename is not None:
                self.writer.write(message)
            print(message)
        self.iter += 1
        return loss

    def net_f(self, x, t):  # 在backward中调用
        output_uv = self.dnn(torch.concat([x, t], dim=1))
        u = output_uv[:, 0:1]
        v = output_uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        # u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        # u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), retain_graph=True, create_graph=True)[0]
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        # v_xxx = torch.autograd.grad(v_xx, x, grad_outputs=torch.ones_like(v_xx), retain_graph=True, create_graph=True)[0]
        # v_xxxx = torch.autograd.grad(v_xxx, x, grad_outputs=torch.ones_like(v_xxx), retain_graph=True, create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]


        fu = u_t + 0.5*v_xx + (u**2 + v**2)*v
        fv = v_t - 0.5*u_xx - (u**2 + v**2)*u  
        
        return fu, fv

    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000):
        self.dnn.train()
        self.iter = init_iter
        self.optimizer_Adam = torch.optim.AdamW(self.dnn.parameters())
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=LBFGS_max_iter, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-7, 
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

    def load_model(self, filename):
        if filename is not None:
            load_path = os.path.join(current_dir, 'model', filename)
            self.dnn.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}.")

    def save_model(self, filename):
        if filename is not None:
            save_path = os.path.join(current_dir, 'model', filename)
            torch.save(self.dnn.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}.")      

    def predict(self, X_star): # 一般不改
        X_star = torch.tensor(X_star).float().to(device)
        self.dnn.eval()
        op_star = self.dnn(X_star)
        u_star = op_star[:, 0:1]
        v_star = op_star[:, 1:2]
        u_star = u_star.detach().cpu().numpy()
        v_star = v_star.detach().cpu().numpy()
        return u_star, v_star

def get_subdomain_points(x, t_min, t_max, Ns=1000):
    """在指定子域内生成采样点"""
    t_sub = np.linspace(t_min, t_max, int(Ns**0.5))
    x_sub = np.linspace(x.min(), x.max(), int(Ns**0.5))
    X_sub, T_sub = np.meshgrid(x_sub, t_sub)
    X_star_sub = np.hstack((X_sub.flatten()[:, None], T_sub.flatten()[:, None]))
    return X_star_sub

if __name__ == "__main__":
    data = scipy.io.loadmat(os.path.join(current_dir, 'data', 'NLS.mat'))
    t = data['tt'].T.flatten()[:,None]
    x = data['x'].T.flatten()[:,None]
    Exact_u = np.real(data['uu']).T
    Exact_v = np.imag(data['uu']).T
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2) 

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.flatten()[:,None]
    v_star = Exact_v.flatten()[:,None]
    h_star = Exact_h.flatten()[:,None]
    
    lb = X_star.min(0)
    ub = X_star.max(0)

    N0 = 50
    Nf = 10000
    iterations = 10000
    layers = [2, 50, 50, 50, 50, 2]

    # 时间分段
    t_segments = [
        [0, 0.39],
        [0.39, 0.79],
        [0.79, 1.17],
        [1.17, 1.57]
    ]
    n_segments = len(t_segments)
    
    models = []
    earliest_solutions = []
    
    # 为每个子域生成采样点
    subdomain_points = []
    for seg in t_segments:
        X_star_sub = get_subdomain_points(x, seg[0], seg[1], Ns=1000)
        subdomain_points.append(X_star_sub)

    for seg_idx, (t_min, t_max) in enumerate(t_segments):
        print(f"\n=== Training Segment {seg_idx} [{t_min}, {t_max}] ===")
        
        t_idx = np.where((t >= t_min) & (t <= t_max))[0]
        
        if seg_idx == 0:
            x0 = X[0:1, :].T
            u0 = Exact_u[0:1, :].T
            v0 = Exact_v[0:1, :].T
            idx = np.random.choice(x0.shape[0], N0, replace=False)
            x0 = x0[idx, :]
            u0 = u0[idx, :]
            v0 = v0[idx, :]
            t0_val = t_min

        else:
            prev_model = models[seg_idx-1]
            t0_val = t_min
            x0 = x
            X0 = np.hstack((x0, np.full_like(x0, t0_val)))
            u0 ,v0= prev_model.predict(X0)
        
        tb = t[t_idx]
        idx_tb = np.random.choice(tb.shape[0], min(50, tb.shape[0]), replace=False)
        tb = tb[idx_tb, :]
        
        Xf = np.hstack((
            np.random.uniform(x.min(), x.max(), (Nf, 1)),
            np.random.uniform(t_min, t_max, (Nf, 1))
        ))
        
        if seg_idx == 0:
            model = PhysicsInformedNN(layers, lb, ub)
            model.set_training_data(x0, u0, v0, tb, Xf)
        else:
            model = IbcPhysicsInformedNN(layers, lb, ub)
            model.set_training_data(x0, u0, v0, tb, Xf, earliest_solutions)
        
        logfilename = f"log-ibc_segment_{seg_idx}.txt"
        model.train(iterations, logfilename)
        
        model_name = f"model_ibc_segment_{seg_idx}.pth"
        model.save_model(model_name)
        models.append(model)
        
        # 转换为 Tensor
        X_ref = subdomain_points[seg_idx]
        u_ref,v_ref = model.predict(X_ref)
        h_ref = np.sqrt(u_ref**2 + v_ref**2)
        X_ref_tensor = torch.tensor(X_ref).float().to(device)
        u_ref_tensor = torch.tensor(u_ref).float().to(device)
        v_ref_tensor = torch.tensor(v_ref).float().to(device)
        h_ref_tensor = torch.tensor(h_ref).float().to(device)
        earliest_solutions.append((X_ref_tensor, h_ref_tensor))

    print("\n=== Combining Solutions from All Segments ===")
    h_pred_combined = np.zeros_like(h_star)

    for seg_idx, (t_min, t_max) in enumerate(t_segments):
        model = models[seg_idx]
        
        t_idx = np.where((t >= t_min) & (t <= t_max))[0]
        seg_points = []
        
        for time_idx in t_idx:
            x_time = X[time_idx, :]
            t_time = T[time_idx, :]
            seg_points.append(np.hstack((x_time.flatten()[:, None], 
                                        t_time.flatten()[:, None])))
        
        X_seg = np.vstack(seg_points)
        u_pred_seg,v_pred_seg  = model.predict(X_seg)
        h_pred_seg = np.sqrt(u_pred_seg**2 + v_pred_seg**2)
        
        seg_mask = ((X_star[:, 1] >= t_min) & (X_star[:, 1] <= t_max))
        h_pred_combined[seg_mask] = h_pred_seg
    
    error_h = np.linalg.norm(h_star - h_pred_combined, 2) / np.linalg.norm(h_star, 2)
    print('Combined Error u: %e' % error_u)
    
    H_pred = griddata(X_star, h_pred_combined.flatten(), (X, T), method='cubic')

    # 可视化
    lb = X_star.min(0)
    ub = X_star.max(0)
    idx_x = np.random.choice(x.shape[0], 100, replace=False)
    x0 = x[idx_x, :]
    idx_t = np.random.choice(t.shape[0], 100, replace=False)
    tb = t[idx_t, :]
    X0 = np.concatenate([x0, 0*x0], axis=1)
    X_lb = np.concatenate([0*tb + lb[0], tb], axis=1)
    X_ub = np.concatenate([0*tb + ub[0], tb], axis=1)
    X_boundary = np.vstack([X0, X_lb, X_ub])

    fig = plt.gcf()
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.1, right=0.9, wspace=0.5)

    ax0 = plt.subplot(gs0[0, 0])
    h0 = ax0.imshow(H_pred.T, interpolation='nearest', cmap='rainbow',
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h0, cax=cax0)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax0.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[100]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[150]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$x$')
    ax0.legend(frameon=False, loc = 'best')
    ax0.set_title('$Pred\ u(t,x)$', fontsize = 10)

    Error = np.abs(Exact_h - H_pred)
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

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[50,:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x, H_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = -1.0$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.01,4.1])
    ax.set_aspect( (5 - (-5)) / (4.1 - 0), adjustable='box' )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[100,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.01,4.1])
    ax.set_aspect( (5 - (-5)) / (4.1 - 0), adjustable='box' )
    ax.set_title('$t = 0$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[150,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[150,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.01,4.1])
    ax.set_aspect( (5 - (-5)) / (4.1 - 0), adjustable='box' )
    ax.set_title('$t = 1.0$', fontsize = 10)

    fig.savefig(os.path.join(current_dir, 'png', 'Ibc test.png'))
    plt.show()
