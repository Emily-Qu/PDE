import os
current_dir = os.path.dirname(os.path.abspath(__file__))
def make_folder(folder_name):
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
make_folder('data')
make_folder('log')
make_folder('model')
make_folder('png')
import warnings
warnings.filterwarnings('ignore')
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:1')
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

# 在代码开头添加训练模式标志
TRAIN_MODE = True  # 设置为True进行训练，False则加载已训练模型进行测试


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
    def __init__(self, layers, lb, ub, prev_model=None):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.dnn = torch.nn.Sequential(Scale(lb, ub), DNN(layers)).to(device)
        
        # 如果提供了前一个模型，则加载其权重
        if prev_model is not None:
            self.dnn.load_state_dict(prev_model.dnn.state_dict())
            print("Loaded weights from previous model")
        else:
            self.dnn.apply(xavier)
    
    def set_training_data(self, x0, u0, tb, Xf, X_prev=None, u_prev=None):
        # 确保所有输入数据都是PyTorch张量
        self.x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
       
        self.tb = torch.tensor(tb, dtype=torch.float32).to(device)
        self.xf = torch.tensor(Xf[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.tf = torch.tensor(Xf[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        
        self.x_lb = torch.full(tb.shape, self.lb[0], dtype=torch.float32).to(device)
        self.x_ub = torch.full(tb.shape, self.ub[0], dtype=torch.float32).to(device)
        # self.t0 = torch.zeros_like(self.x0, dtype=torch.float32).to(device)
        self.t0 = torch.full_like(self.x0, -0.5).to(device)

        
        # 存储前一个窗口的重叠点数据
        if X_prev is not None:
            self.X_prev = torch.tensor(X_prev, dtype=torch.float32).to(device)
            self.u_prev = torch.tensor(u_prev, dtype=torch.float32).to(device)
            
        else:
            self.X_prev = None
    
    def backward(self):
        # 初始条件损失
        u0_pred = self.dnn(torch.cat([self.x0, self.t0], dim=1))
        
        loss0 = torch.mean(torch.square(u0_pred - self.u0))
                

        # 边界条件损失
        u_lb_pred = self.dnn(torch.cat([self.x_lb, self.tb], dim=1))
        u_ub_pred = self.dnn(torch.cat([self.x_ub, self.tb], dim=1))
        
        
        loss_b = torch.mean(torch.square(u_lb_pred - u_ub_pred))  

        # PDE残差损失
        f_u = self.net_f(self.xf, self.tf)
        loss_f = torch.mean(torch.square(f_u)) 
        
        # 重叠区域损失（新增）
        loss_prev = 0
        if self.X_prev is not None and self.X_prev.numel() > 0:
            u_prev_pred = self.dnn(self.X_prev)
            
            loss_prev = torch.mean(torch.square(u_prev_pred - self.u_prev)) 
                        
            if self.iter % 100 == 0:
                print(f"Overlap loss: {loss_prev.item():.5e}")

        # 总损失 = 初始条件损失 + 边界损失 + PDE损失 + 重叠区域损失
        loss = loss0 + loss_b + loss_f + loss_prev
        loss.backward()

        if self.iter % 100 == 0:
            message = 'Iter: %d, Loss: %.5e, Loss0: %.5e, Loss_b: %.5e, Loss_f: %.5e, Loss_prev: %.5e' % \
                     (self.iter, loss.item(), loss0.item(), loss_b.item(), loss_f.item(), loss_prev)
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
        u_xxxxxx = torch.autograd.grad(u_xxxxx, x, grad_outputs=torch.ones_like(u_xxxxx), retain_graph=True, create_graph=True)[0]
        u_xxxxxxx = torch.autograd.grad(u_xxxxxx, x, grad_outputs=torch.ones_like(u_xxxxxx), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        return  u_t + (5/98)*u**3*u_x + (5/14)*u_x**3 + (10/7)*u*u_x*u_xx + (5/14)*u**2*u_xxx + 5*u_xx*u_xxx + 3*u_x*u_xxxx + u*u_xxxxx + u_xxxxxxx

    def train(self, nIter, logfilename=None, init_iter=0, LBFGS_max_iter=50000):
        self.dnn.train()
        self.iter = init_iter
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1, 
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

    def predict(self, X_star):
        X_star = torch.tensor(X_star, dtype=torch.float32).to(device)
        self.dnn.eval()
        with torch.no_grad():
            u_star = self.dnn(X_star).cpu().numpy()
            
        return u_star





if __name__ == "__main__":
    # 加载数据
    data = scipy.io.loadmat(os.path.join(current_dir, 'data', '73[-0.5,0.5]201*512.mat'))
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    
    
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.flatten()[:, None]
    
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # 设置参数
    N0 = 100
    Nb = 100
    Nf = 20000
    n_overlap = 50  # 重叠区域采样点数
    layers = [2, 50, 50, 50, 50, 1]
    
    # 滑动窗口参数
    total_time = ub[1] - lb[1]
    num_windows = 3
    window_size = total_time * 0.5
    step_size = window_size * 0.5
    
    # 存储每个窗口的结果
    all_models = []
    prev_model = None
    
    # 初始化数组用于存储拼接后的预测结果
    full_u_pred = np.zeros_like(u_star)
    
    
    # 标记哪些点已经被预测
    predicted_mask = np.zeros(len(X_star), dtype=bool)
    
    for win_idx in range(num_windows):
        print(f"\n===== Processing Window {win_idx+1}/{num_windows} =====")
        
        # 计算当前窗口的时间范围
        t_min = lb[1] + win_idx * step_size
        t_max = min(t_min + window_size, ub[1])
        
        # 当前窗口的lb和ub
        win_lb = [lb[0], t_min]
        win_ub = [ub[0], t_max]
        
        # 模型文件名
        model_name = f"sw73_{win_idx+1}.pth"
        model_path = os.path.join(current_dir, 'model', model_name)
        
        if TRAIN_MODE:
            # 训练模式 - 训练并保存模型
            print("Training mode - Training new model...")
            
            # 准备初始条件
            if win_idx == 0:
                ### 第一个窗口使用原始初始条件
                u0 = Exact_u[0, :].reshape(-1, 1)
                
                idx = np.random.choice(len(x), N0, replace=False)
                x0 = x[idx]
                u0 = u0[idx]
                
                
                # 第一个窗口没有前一个模型
                X_prev, u_prev = None, None
            else:
                # 后续窗口使用前一个窗口在t=t_min时刻的解作为初始条件
                X_init = np.hstack((x, np.ones_like(x)*t_min))
                u_pred = prev_model.predict(X_init)
                
                idx = np.random.choice(len(x), N0, replace=False)
                x0 = x[idx]
                u0 = u_pred[idx]
                
                
                # 计算与前一个窗口的重叠区域
                prev_t_min = lb[1] + (win_idx-1) * step_size
                prev_t_max = min(prev_t_min + window_size, ub[1])
                
                overlap_min = max(prev_t_min, t_min)
                overlap_max = min(prev_t_max, t_max)
                
                # 在重叠区域内随机采样点
                if overlap_max > overlap_min:
                    X_prev = np.zeros((n_overlap, 2))
                    X_prev[:, 0] = lb[0] + (ub[0]-lb[0]) * np.random.rand(n_overlap)
                    X_prev[:, 1] = overlap_min + (overlap_max - overlap_min) * np.random.rand(n_overlap)
                    
                    u_prev = prev_model.predict(X_prev)
                    print(f"Sampled {n_overlap} points in overlap region: t=[{overlap_min:.2f}, {overlap_max:.2f}]")
                else:
                    X_prev, u_prev = None, None
                    print("No overlap with previous window")
            
            # 准备边界条件
            tb = t_min + (t_max - t_min) * lhs(1, Nb)
            
            # 准备区域内的配点
            Xf = np.zeros((Nf, 2))
            Xf[:, 0] = win_lb[0] + (win_ub[0]-win_lb[0]) * lhs(1, Nf).flatten()
            Xf[:, 1] = t_min + (t_max - t_min) * lhs(1, Nf).flatten()
            
            # 创建并训练模型
            model = PhysicsInformedNN(layers, win_lb, win_ub, prev_model)
            logfilename = f"sw73_{win_idx+1}.txt"
            
            model.set_training_data(x0, u0,  tb, Xf, X_prev, u_prev)
            model.train(10000, logfilename)
            
            # 保存当前窗口的模型
            model.save_model(model_name)
            print(f"Model saved to {model_path}")
            
        else:
            # 测试模式 - 加载已训练模型
            print(f"Test mode - Loading model from {model_path}")
            model = PhysicsInformedNN(layers, win_lb, win_ub)
            model.load_model(model_name)
        
        # 将当前模型设为前一个模型（用于下一个窗口）
        prev_model = model
        all_models.append(model)
        
        # 使用当前模型预测整个时间范围
        u_pred = model.predict(X_star)
        
        
        # 确定当前窗口的时间范围对应的点
        window_mask = (X_star[:, 1] >= t_min) & (X_star[:, 1] <= t_max)
        
        # 只更新当前窗口时间范围内的点
        full_u_pred[window_mask] = u_pred[window_mask]
        
        
        # 更新预测掩码
        predicted_mask[window_mask] = True
    
    # 检查是否有未被覆盖的点
    if not np.all(predicted_mask):
        print(f"Warning: {np.sum(~predicted_mask)} points were not predicted by any window.")
        missing_mask = ~predicted_mask
        u_missing = prev_model.predict(X_star[missing_mask])
        
        
        full_u_pred[missing_mask] = u_missing
        
    
    # 计算整体误差
    error_u = np.linalg.norm(u_star - full_u_pred, 2) / np.linalg.norm(u_star, 2)
    
    print('Overall Error u: %e' % (error_u))
    
    
    # 可视化拼接后的结果
    Full_U_pred = griddata(X_star, full_u_pred.flatten(), (X, T), method='cubic')
    
    Full_Error = np.abs(Exact_u - Full_U_pred)
    
    # 绘图代码
    # 创建一个大的图形对象




    # 创建一个大的图形对象
    fig = plt.figure(figsize=(10, 10))

## 创建顶层网格布局：2行1列
    gs_top = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], figure=fig)

## 第一行：热图和窗口图（2x2网格）
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_top[0], wspace=0.3, hspace=0.3)

## 第二行：切片图（1x3网格）
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_top[1], wspace=0.3)

###第一行第一列：预测热图
    ax1 = fig.add_subplot(gs_row1[0])
    im1 = ax1.imshow(Full_U_pred.T, cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax1.set_title('Combined Prediction $|u(t,x)|$')

# 第一行第二列：误差热图
    ax2 = fig.add_subplot(gs_row1[1])
    im2 = ax2.imshow(Full_Error.T, cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x$')
    ax2.set_title('Combined Prediction Error')

# 第一行第三列（跨两行）：窗口划分图
    ax3 = fig.add_subplot(gs_row1[2])  # 跨所有行
    colors = ['red', 'green', 'blue']
    for i in range(num_windows):
        t_min = lb[1] + i * step_size
        t_max = min(t_min + window_size, ub[1])
        ax3.axvline(x=t_min, color=colors[i], linestyle='--', alpha=0.7)
        ax3.axvline(x=t_max, color=colors[i], linestyle='--', alpha=0.7)
        ax3.axvspan(t_min, t_max, alpha=0.1, color=colors[i], label=f'Window {i+1}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Window')
    ax3.set_title('Time Windows with Overlap')
    ax3.legend()

# 第二行：切片图
    ax4 = fig.add_subplot(gs_row2[0])
    ax4.plot(x, Exact_u[50,:], 'b-', linewidth=2, label='Exact')
    ax4.plot(x, Full_U_pred[50,:], 'r--', linewidth=2, label='Prediction')
    ax4.set_xlabel('$x$')
    ax4.set_ylabel('$|u(t,x)|$')
    ax4.set_title('$t = -0.25$', fontsize=12)
    ax4.set_xlim([-15.1,15.1])
    ax4.set_ylim([-0.01,16])
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_aspect((15 - (-15)) / (16 - 0), adjustable='box')

    ax5 = fig.add_subplot(gs_row2[1])
    ax5.plot(x, Exact_u[100,:], 'b-', linewidth=2, label='Exact')       
    ax5.plot(x, Full_U_pred[100,:], 'r--', linewidth=2, label='Prediction')
    ax5.set_xlabel('$x$')
    ax5.set_ylabel('$|u(t,x)|$')
    ax5.set_title('$t = 0$', fontsize=12)
    ax5.set_xlim([-15.1,15.1])
    ax5.set_ylim([-0.01,16])
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.set_aspect((15 - (-15)) / (16 - 0), adjustable='box')
    ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    ax6 = fig.add_subplot(gs_row2[2])
    ax6.plot(x, Exact_u[150,:], 'b-', linewidth=2, label='Exact')       
    ax6.plot(x, Full_U_pred[150,:], 'r--', linewidth=2, label='Prediction')
    ax6.set_xlabel('$x$')
    ax6.set_ylabel('$|u(t,x)|$')
    ax6.set_title('$t = 0.25$', fontsize=12)
    ax6.set_xlim([-15.1,15.1])
    ax6.set_ylim([-0.01,16])
    ax6.grid(True, linestyle='--', alpha=0.5)
    ax6.set_aspect((15 - (-15)) / (16 - 0), adjustable='box')

    
    plt.tight_layout()
    
    ######根据模式保存不同名称的图片
    if TRAIN_MODE:
        plt.savefig(os.path.join(current_dir, 'png', 'sw-73-train.png'))
    else:
        plt.savefig(os.path.join(current_dir, 'png', 'sw-73-loaded.png'))
    plt.show()
