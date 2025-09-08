import sys
# 若需要使用自定义绘图函数 newfig, savefig，可修改路径
sys.path.insert(0, '/home/jovyan/PINNs-master/Utilities/')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 读取pkl文件中的u1数据
pkl_file = '/home/jovyan/bc-PINN-main/fifthkdv_0001_ICGL_TL/Sequential_training_150_201/u1.pkl'
with open(pkl_file, 'rb') as f:
    u1 = pickle.load(f)
data = scipy.io.loadmat('/home/jovyan/bc-PINN-main/data/fifth_kdv_exact.mat')
    
t = data['tt'].flatten()[:,None]
x = data['xx'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact).T

X, T = np.meshgrid(x,t)
    
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]
error_u = np.linalg.norm(u_star-u1,2)/np.linalg.norm(u_star,2)
print("u1 shape:", u1.shape)
print("u_star shape:", u_star.shape)
print("Exact_u shape:", Exact_u.shape)
print('Error u: %e' % (error_u))



# 构造网格

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# 如果 u1 的数据是按照 X_star 的顺序排列的
# 则利用 griddata 将散点数据转换为二维矩阵
U_pred = griddata(X_star, u1.flatten(), (X, T), method='cubic')


lb = np.array([-5.0, 0.0])
ub = np.array([5.0, 2.0])
idx_x = np.random.choice(x.shape[0], 100, replace=False)
x0 = x[idx_x, :]
idx_t = np.random.choice(t.shape[0], 100, replace=False)
tb = t[idx_t, :]

X0 = np.concatenate((x0, 0*x0), 1)      # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])


# 绘图：二维色彩图与时间切片对比
fig, ax = newfig(1.0, 0.9)
ax.axis('off')
    
####### Row 0: u(x,t) 的二维色彩图 ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax0 = plt.subplot(gs0[:, :])
    
h = ax0.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu', 
                   extent=[lb[1], ub[1], lb[0], ub[0]], 
                   origin='lower', aspect='auto')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
    
ax0.plot(X_u_train[:,1], X_u_train[:,0], 'kx', 
             label='Data (%d points)' % (X_u_train.shape[0]), markersize=4, clip_on=False)
    
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax0.plot(t[50]*np.ones((2,1)), line, 'k--', linewidth=1)
ax0.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth=1)
ax0.plot(t[150]*np.ones((2,1)), line, 'k--', linewidth=1)
    
ax0.set_xlabel('$t$')
ax0.set_ylabel('$x$')
leg = ax0.legend(frameon=False, loc='best')
ax0.set_title('$u(x,t)$', fontsize=10)
    
####### Row 1: u(x,t) 切片对比 ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_u[:,50], 'b-', linewidth=2, label='Exact')       
ax.plot(x, U_pred[50,:], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x,t)$')    
ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-1.1, 1.1])


# 即 ax.set_aspect( 10/0.3 )，约为 33.33

    
ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_u[:,100], 'b-', linewidth=2, label='Exact')       
ax.plot(x, U_pred[100,:], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x,t)$')
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-1.1, 1.1])

# 即 ax.set_aspect( 10/0.3 )，约为 33.33

ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_u[:,150], 'b-', linewidth=2, label='Exact')       
ax.plot(x, U_pred[150,:], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x,t)$')
ax.axis('square')
ax.set_xlim([-5.1,5.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = %.2f$' % (t[150]), fontsize=10)

# 即 ax.set_aspect( 10/0.3 )，约为 33.33

    
savefig('/home/jovyan/bc-PINN-main/figures/bc-fifthkdv')
plt.show()
