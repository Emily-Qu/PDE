import numpy as np
import matplotlib.pyplot as plt

# 定义交换 x <-> t 后的解 psi2_swapped(t,x)
def psi2_swapped(x, t):
    N = np.exp(12*t) \
        - 1j*np.exp(4j*x + 4*(1+1j)*t) \
        + np.exp(4*(1+1j)*x + 4*(2+1j)*t) \
        - 1j*np.exp(4*x)
    
    D = np.cos(4*(x+t)) - np.cosh(4*(x-t)) - 2*np.cosh(8*t)
    
    psi = 2*(1-1j) * (N/D) * np.exp(-2*(1+1j)*(x+3*t))
    return psi

# 网格 (t 横轴, x 纵轴)
t = np.linspace(-1, 1, 300)
x = np.linspace(-5, 5, 500)
X, T = np.meshgrid(x, t)

# 计算解
Psi = psi2_swapped(X, T)
u = np.real(Psi)   # 实部
v = np.imag(Psi)   # 虚部

# 保存到 npz 文件
np.savez("psi2_uv.npz", x=x, t=t, u=u, v=v)

# 测试读取
data = np.load("psi2_uv.npz")
print(data.files)   # ['x', 't', 'u', 'v']
print(data["u"].shape, data["v"].shape)

# 可视化：实部和虚部
# 取模值可视化
plt.figure(figsize=(8,6))
plt.pcolormesh(X, T, np.abs(Psi), shading='auto', cmap='viridis')
plt.colorbar(label=r"$|\psi_2(x,t)|$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Visualization of $|\psi_2(x,t)|$")
plt.savefig("psi2_solution.png", dpi=300, bbox_inches='tight')
plt.show()
