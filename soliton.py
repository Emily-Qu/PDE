
# 5阶热力图

import numpy as np
import matplotlib.pyplot as plt

# 定义 f(x,t)
def f(x, t, k1, k2):
    A = np.exp(k1 * (x - k1**2 * t))
    B = np.exp(k2 * (x - k2**2 * t))
    C_num = (k1 - k2)**2 * (k1**2 - k1*k2 + k2**2)
    C_den = (k1 + k2)**2 * (k1**2 + k1*k2 + k2**2)
    C = C_num / C_den
    D = np.exp((k1 + k2)*x - (k1**3 + k2**3)*t)
    return 1 + A + B + C * D

# 计算 u(x,t) = d²(ln f)/dx²
def compute_u_surface(x, t, k1, k2):
    dx = x[1] - x[0]
    X, T = np.meshgrid(x, t)
    F = f(X, T, k1, k2)
    logF = np.log(F)
    u = np.zeros_like(logF)
    u[:, 1:-1] = (logF[:, 2:] - 2 * logF[:, 1:-1] + logF[:, :-2]) / dx**2
    return u

# 参数设置
x = np.linspace(-25, 25, 300)
t = np.linspace(-25, 25, 200)
k1, k2 = 1.0,1.4

u = compute_u_surface(x, t, k1, k2)

# 热图绘制
plt.figure(figsize=(10, 6))
extent = [x.min(), x.max(), t.min(), t.max()]
plt.imshow(u, extent=extent, origin='lower', aspect='auto', cmap='rainbow')
plt.colorbar(label='u(x, t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'KdV 二孤子解 u(x,t) 热图，k1={k1}, k2={k2}')

# 保存图片
plt.savefig('5kdv_2solheat.png', dpi=300)

# 显示图像
plt.tight_layout()
plt.show()




# # 5阶三维图

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 定义 f(x,t)
# def f(x, t, k1, k2):
#     A = np.exp(k1 * (x - k1**2 * t))
#     B = np.exp(k2 * (x - k2**2 * t))
#     C_num = (k1 - k2)**2 * (k1**2 - k1*k2 + k2**2)
#     C_den = (k1 + k2)**2 * (k1**2 + k1*k2 + k2**2)
#     C = C_num / C_den
#     D = np.exp((k1 + k2)*x - (k1**3 + k2**3)*t)
#     return 1 + A + B + C * D

# # 计算 u(x,t) = d²(ln f)/dx²
# def compute_u_surface(x, t, k1, k2):
#     dx = x[1] - x[0]
#     X, T = np.meshgrid(x, t)
#     F = f(X, T, k1, k2)
#     logF = np.log(F)

#     # 使用中心差分计算 d²(ln f)/dx²
#     u = np.zeros_like(logF)
#     u[:, 1:-1] = (logF[:, 2:] - 2 * logF[:, 1:-1] + logF[:, :-2]) / dx**2

#     return X, T, u

# # 参数设置
# x = np.linspace(-25, 25, 300)
# t = np.linspace(-25, 25, 200)
# k1, k2 = 1.0, 1.5  # 你可以修改为其他参数

# X, T, U = compute_u_surface(x, t, k1, k2)

# # 绘图
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, T, U, cmap='viridis')

# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# ax.set_title(f'KdV 二孤子解 u(x,t)，k1={k1}, k2={k2}')

# # 保存图像（分辨率 300 dpi）
# plt.savefig('52.png', dpi=600)

# # 显示图像
# plt.tight_layout()
# plt.show()



# ## 5阶密度图
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义 f(x,t)
# def f(x, t, k1, k2):
#     A = np.exp(k1 * (x - k1**4 * t))
#     B = np.exp(k2 * (x - k2**4 * t))
#     C_num = (k1 - k2)**2 * (k1**2 - k1*k2 + k2**2)
#     C_den = (k1 + k2)**2 * (k1**2 + k1*k2 + k2**2)
#     C = C_num / C_den
#     D = np.exp((k1 + k2)*x - (k1**5 + k2**5)*t)
#     return 1 + A + B + C * D

# # 计算 u(x,t) = d²(ln f)/dx²
# def compute_u_surface(x, t, k1, k2):
#     dx = x[1] - x[0]
#     X, T = np.meshgrid(x, t)
#     F = f(X, T, k1, k2)
#     logF = np.log(F)
#     u = np.zeros_like(logF)
#     u[:, 1:-1] = (logF[:, 2:] - 2 * logF[:, 1:-1] + logF[:, :-2]) / dx**2
#     return X, T, u

# # 设置参数
# x = np.linspace(-15, 15, 300)
# t = np.linspace(-2, 2, 200)
# k1, k2 = 1.0, 1.4

# X, T, u = compute_u_surface(x, t, k1, k2)

# # 绘制密度图（等高填色图）
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(X, T, u, levels=100, cmap='plasma')
# plt.colorbar(contour, label='u(x,t)')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'KdV 二孤子解 u(x,t) 密度图，k1={k1}, k2={k2}')

# # 保存图像
# plt.savefig('52密度.png', dpi=300)

# plt.tight_layout()
# plt.show()




#  ##    7阶密度图
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义 f(x,t)
# def f(x, t, k1, k2):
#     A = np.exp(k1 * (x - k1**6 * t))
#     B = np.exp(k2 * (x - k2**6 * t))
#     C_num = (k1 - k2)**2 
#     C_den = (k1 + k2)**2 
#     C = C_num / C_den
#     D = np.exp((k1 + k2)*x - (k1**7 + k2**7)*t)
#     return 1 + A + B + C * D

# # 计算 u(x,t) = d²(ln f)/dx²
# def compute_u_surface(x, t, k1, k2):
#     dx = x[1] - x[0]
#     X, T = np.meshgrid(x, t)
#     F = f(X, T, k1, k2)
#     logF = np.log(F)
#     u = np.zeros_like(logF)
#     u[:, 1:-1] = (logF[:, 2:] - 2 * logF[:, 1:-1] + logF[:, :-2]) / dx**2
#     return X, T, u

# # 设置参数
# x = np.linspace(-20, 20, 300)
# t = np.linspace(0, 2, 200)
# k1, k2 = 1.0,1.4

# X, T, u = compute_u_surface(x, t, k1, k2)

# # 绘制密度图（等高填色图）
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(X, T, u, levels=100, cmap='plasma')
# plt.colorbar(contour, label='u(x,t)')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'KdV 二孤子解 u(x,t) 密度图，k1={k1}, k2={k2}')

# # 保存图像
# plt.savefig('72密度.png', dpi=300)

# plt.tight_layout()
# plt.show()



# # 定义 f(x,t)
# def f(x, t, k1, k2):
#     A = np.exp(k1 * (x - k1**6 * t))
#     B = np.exp(k2 * (x - k2**6 * t))
#     C_num = (k1 - k2)**2 
#     C_den = (k1 + k2)**2 
#     C = C_num / C_den
#     D = np.exp((k1 + k2)*x - (k1**7 + k2**7)*t)
#     return 1 + A + B + C * D





# # 7阶三维图

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 定义 f(x,t)
# def f(x, t, k1, k2):
#     A = np.exp(k1 * (x - k1**6 * t))
#     B = np.exp(k2 * (x - k2**6 * t))
#     C_num = (k1 - k2)**2 
#     C_den = (k1 + k2)**2 
#     C = C_num / C_den
#     D = np.exp((k1 + k2)*x - (k1**7 + k2**7)*t)
#     return 1 + A + B + C * D

# # 计算 u(x,t) = d²(ln f)/dx²
# def compute_u_surface(x, t, k1, k2):
#     dx = x[1] - x[0]
#     X, T = np.meshgrid(x, t)
#     F = f(X, T, k1, k2)
#     logF = np.log(F)

#     # 使用中心差分计算 d²(ln f)/dx²
#     u = np.zeros_like(logF)
#     u[:, 1:-1] = (logF[:, 2:] - 2 * logF[:, 1:-1] + logF[:, :-2]) / dx**2
#     # u = 28*u

#     return X, T, u

# # 参数设置
# x = np.linspace(-25, 25, 300)
# t = np.linspace(0, 2, 200)
# k1, k2 = 1.0, 0.9  # 你可以修改为其他参数

# X, T, U = compute_u_surface(x, t, k1, k2)

# # 绘图
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, T, U, cmap='viridis')

# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# ax.set_title(f'KdV 二孤子解 u(x,t)，k1={k1}, k2={k2}')

# # 保存图像（分辨率 300 dpi）
# plt.savefig('7kdv_solution.png', dpi=300)

# # 显示图像
# plt.tight_layout()
# plt.show()
